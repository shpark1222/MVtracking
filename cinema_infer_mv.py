from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_convvit_class():
    try:
        cinema_pkg = importlib.import_module("cinema")
        if hasattr(cinema_pkg, "ConvViT"):
            return cinema_pkg.ConvViT
    except Exception:
        pass

    module_candidates = [
        "cinema.convvit",
        "cinema.conv_vit",
        "cinema.models.convvit",
        "cinema.models.conv_vit",
        "cinema.model.convvit",
        "cinema.model.conv_vit",
    ]

    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(mod, "ConvViT"):
            return mod.ConvViT

    raise ImportError(
        "ConvViT class not found. "
        "Check that CineMA is installed in the selected conda env and that import cinema works."
    )


def _load_input(path: Path) -> Tuple[np.ndarray, str]:
    data = np.load(path, allow_pickle=True)
    if "cine" not in data or "view" not in data:
        raise ValueError("input.npz must contain 'cine' and 'view'")
    cine = np.asarray(data["cine"], dtype=np.float32)
    view = data["view"]
    if isinstance(view, np.ndarray):
        view = view.item()
    view = str(view)
    if cine.ndim != 3:
        raise ValueError(f"cine must have shape (T,H,W), got {cine.shape}")
    return cine, view


def _normalize_view(view: str) -> str:
    view = view.strip()
    if view in ("lax_2c", "lax_4c"):
        return view
    if view in ("lax_3c", "lax_3ch", "3ch"):
        return "lax_4c"
    raise ValueError(f"Unsupported view: {view}")


def _normalize_frame_percentile(frame: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float32)
    lo, hi = np.percentile(f, [p_lo, p_hi])
    if hi > lo:
        f = np.clip(f, lo, hi)
        f = (f - lo) / (hi - lo)
        return f.astype(np.float32, copy=False)
    return np.zeros_like(f, dtype=np.float32)


def run_inference(
    cine: np.ndarray,
    view: str,
    debug_dir: str | None = None,
    debug_frame_idx: int = 0,
    debug_all_frames: bool = False,
) -> dict:
    view = _normalize_view(view)
    convvit_cls = _resolve_convvit_class()
    model = convvit_cls.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/landmark_coordinate/{view}/{view}_0.safetensors",
        config_filename=f"finetuned/landmark_coordinate/{view}/config.yaml",
    )
    model.to("cpu")
    model = model.float()
    model.eval()

    _, H, W = cine.shape
    ALIGN = 16
    pad_h = (ALIGN - (H % ALIGN)) % ALIGN
    pad_w = (ALIGN - (W % ALIGN)) % ALIGN

    Hp = H + pad_h
    Wp = W + pad_w
    scale = np.array([Wp, Hp, Wp, Hp, Wp, Hp], dtype=np.float32)

    coords_list: list[np.ndarray] = []

    with torch.no_grad():
        for t in range(cine.shape[0]):
            frame = np.asarray(cine[t], dtype=np.float32)
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w)), mode="constant")
            frame = _normalize_frame_percentile(frame, 1.0, 99.0)

            tensor = torch.from_numpy(frame).to(dtype=torch.float32)[None, None, ...]
            coords = model({view: tensor})[0]
            if isinstance(coords, torch.Tensor):
                coords = coords.detach().cpu().numpy()
            coords = np.asarray(coords, dtype=np.float32).reshape(-1)
            if coords.shape[0] < 6:
                raise RuntimeError(f"Invalid landmark output shape: {coords.shape}")

            coords = coords[:6] * scale
            coords = coords.reshape(3, 2)[:, ::-1].reshape(-1)

            coords[0::2] = np.clip(coords[0::2], 0, Wp - 1)
            coords[1::2] = np.clip(coords[1::2], 0, Hp - 1)

            if debug_dir is not None and (debug_all_frames or t == debug_frame_idx):
                outdir = Path(debug_dir)
                outdir.mkdir(parents=True, exist_ok=True)

                x1, y1, x2, y2, x3, y3 = coords.tolist()
                plt.figure()
                plt.imshow(frame, cmap="gray")
                plt.scatter([x1, x2, x3], [y1, y2, y3], s=60)
                plt.title(f"{view} t={t} (Hp={Hp}, Wp={Wp})")
                plt.axis("off")
                plt.savefig(outdir / f"overlay_{view}_t{t:03d}.png", dpi=150, bbox_inches="tight")
                plt.close()

            coords_list.append(coords)

    coords_all = np.stack(coords_list, axis=1)
    mv_coords = coords_all[0:4, :]
    return {
        "view": view,
        "coords": coords_all.tolist(),
        "mv": mv_coords.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--debug_frame_idx", type=int, default=0)
    parser.add_argument("--debug_all_frames", action="store_true")
    args = parser.parse_args()

    cine, view = _load_input(Path(args.input))
    output = run_inference(
        cine,
        view,
        debug_dir=args.debug_dir,
        debug_frame_idx=args.debug_frame_idx,
        debug_all_frames=args.debug_all_frames,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
