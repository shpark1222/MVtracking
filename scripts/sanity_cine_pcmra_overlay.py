import argparse
import importlib
import importlib.util

import numpy as np
from scipy.ndimage import map_coordinates

from mvpack_io import load_mvpack_h5

_imageio_spec = importlib.util.find_spec("imageio.v2")
imageio = importlib.import_module("imageio.v2") if _imageio_spec else None


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    vmin = float(np.nanmin(img))
    vmax = float(np.nanmax(img))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(img, dtype=np.float64)
    return np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)


def _cine_frame(cine: np.ndarray, t: int) -> np.ndarray:
    if cine.ndim == 2:
        return cine.astype(np.float32)
    nt = cine.shape[2]
    idx = int(np.clip(t, 0, max(nt - 1, 0)))
    return cine[:, :, idx].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check cine/pcmra alignment using edges.")
    parser.add_argument("mvpack", help="Path to mvpack.h5")
    parser.add_argument("cine_key", help="Cine key (e.g. 2ch)")
    parser.add_argument("--t", type=int, default=0, help="Time index (default: 0)")
    parser.add_argument("--out", default="cine_pcmra_overlay.png", help="Output PNG path")
    args = parser.parse_args()

    if imageio is None:
        print("imageio.v2 is not available; cannot write PNG output.")
        return 1

    pack = load_mvpack_h5(args.mvpack)
    if args.cine_key not in pack.cine_planes:
        raise KeyError(f"cine key not found: {args.cine_key}")

    cine_plane = pack.cine_planes[args.cine_key]
    cine = cine_plane["img"]
    cine_geom = cine_plane["geom"]
    edges = cine_geom.edges
    if edges is None:
        raise RuntimeError("cine edges missing; rebuild mvpack with cine edges.")

    cine_frame = _cine_frame(cine, args.t)
    H, W = cine_frame.shape

    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack(
        [
            ii.astype(np.float64),
            jj.astype(np.float64),
            np.zeros_like(ii, dtype=np.float64),
            np.ones_like(ii, dtype=np.float64),
        ],
        axis=0,
    ).reshape(4, -1)

    edges = np.asarray(edges, dtype=np.float64)
    if edges.shape == (3, 4):
        edges = np.vstack([edges, np.array([0.0, 0.0, 0.0, 1.0])])
    XYZ = (edges @ coords)[:3, :].T

    A = pack.geom.A
    orgn4 = pack.geom.orgn4
    abc = np.linalg.solve(A, (XYZ - orgn4).T)
    rowq = abc[1].reshape(H, W)
    colq = abc[0].reshape(H, W)
    slcq = abc[2].reshape(H, W)

    pcmra3d = pack.pcmra[:, :, :, int(np.clip(args.t, 0, pack.pcmra.shape[3] - 1))]
    sample_coords = np.vstack([rowq.ravel(), colq.ravel(), slcq.ravel()])
    pcmra_slice = map_coordinates(
        pcmra3d,
        sample_coords,
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(H, W)

    cine_norm = _normalize(cine_frame)
    pcmra_norm = _normalize(pcmra_slice)

    rgb = np.stack([cine_norm, cine_norm, cine_norm], axis=-1)
    rgb[..., 0] = np.clip(rgb[..., 0] + pcmra_norm, 0.0, 1.0)

    imageio.imwrite(args.out, (rgb * 255.0).astype(np.uint8))
    print(f"Overlay saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
