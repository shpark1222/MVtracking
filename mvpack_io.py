import json
import os
import glob
from dataclasses import dataclass
from typing import Dict, Optional

import h5py
import numpy as np


@dataclass
class CineGeom:
    ipp: np.ndarray
    iop: np.ndarray
    ps: np.ndarray
    axis_map: Optional[Dict[str, np.ndarray]] = None


@dataclass
class VolGeom:
    orgn4: np.ndarray
    A: np.ndarray
    axis_map: Optional[Dict[str, np.ndarray]] = None
    iop: Optional[np.ndarray] = None
    ipp0: Optional[np.ndarray] = None
    pixel_spacing: Optional[np.ndarray] = None
    slice_positions: Optional[np.ndarray] = None
    slice_order: Optional[np.ndarray] = None


@dataclass
class MVPack:
    cine_planes: Dict[str, dict]  # key -> {"img": (Ny,Nx,NtC), "geom": CineGeom}
    pcmra: np.ndarray  # (Ny,Nx,Nz,Nt)
    vel: np.ndarray  # (Ny,Nx,Nz,3,Nt)
    geom: VolGeom
    ke: Optional[np.ndarray] = None  # (Ny,Nx,Nz,Nt)
    vortmag: Optional[np.ndarray] = None  # (Ny,Nx,Nz,Nt)


def _read_ds(f: h5py.File, path: str) -> np.ndarray:
    return np.array(f[path][()])


def _read_axis_map_json(group) -> Optional[Dict[str, np.ndarray]]:
    if group is None:
        return None
    payload = group.attrs.get("axis_map_json")
    if payload is None:
        return None
    try:
        data = json.loads(payload)
    except Exception:
        return None
    out: Dict[str, np.ndarray] = {}
    for key in ("Rows", "Columns", "Slices"):
        if key in data:
            out[key] = np.asarray(data[key], dtype=np.float64).reshape(3)
    return out if out else None


def find_mvpack_in_folder(folder: str) -> str:
    """
    우선순위:
      1) folder/mvpack.h5
      2) folder/**/mvpack.h5 (바로 아래는 아니어도)
      3) folder 안의 *.h5 중 이름에 mvpack 포함
    """
    p0 = os.path.join(folder, "mvpack.h5")
    if os.path.exists(p0):
        return p0

    hits = glob.glob(os.path.join(folder, "**", "mvpack.h5"), recursive=True)
    if hits:
        return hits[0]

    hits2 = glob.glob(os.path.join(folder, "*.h5"))
    for p in hits2:
        if "mvpack" in os.path.basename(p).lower():
            return p

    raise FileNotFoundError(f"mvpack.h5 not found under: {folder}")


def _coerce_cine_axes(img: np.ndarray, ny: int, nx: int) -> np.ndarray:
    if img.ndim != 3:
        return img

    shape = img.shape
    for t_axis in range(3):
        spatial_axes = [ax for ax in range(3) if ax != t_axis]
        if shape[spatial_axes[0]] == ny and shape[spatial_axes[1]] == nx:
            return np.transpose(img, (spatial_axes[0], spatial_axes[1], t_axis)).copy()
        if shape[spatial_axes[0]] == nx and shape[spatial_axes[1]] == ny:
            return np.transpose(img, (spatial_axes[1], spatial_axes[0], t_axis)).copy()

    if shape[0] < shape[1] and shape[0] < shape[2]:
        return np.transpose(img, (1, 2, 0)).copy()

    return img


def load_mvpack_h5(h5_path: str) -> MVPack:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    def _first_existing_path(f: h5py.File, candidates):
        for p in candidates:
            if p in f:
                return p
        return None

    def _read_attr_float(group, keys, default):
        for k in keys:
            if group is not None and k in group.attrs:
                try:
                    return float(group.attrs.get(k))
                except Exception:
                    pass
        return float(default)

    with h5py.File(h5_path, "r") as f:
        if "/geom/orgn4" not in f or "/geom/A" not in f:
            raise RuntimeError("mvpack.h5 missing /geom/orgn4 or /geom/A")

        orgn4 = _read_ds(f, "/geom/orgn4").reshape(3).astype(np.float64)
        A = _read_ds(f, "/geom/A").astype(np.float64)
        geom_axis_map = _read_axis_map_json(f["/geom"]) if "/geom" in f else None
        geom_iop = _read_ds(f, "/geom/IOP").reshape(6).astype(np.float64) if "/geom/IOP" in f else None
        geom_ipp0 = _read_ds(f, "/geom/IPP0").reshape(3).astype(np.float64) if "/geom/IPP0" in f else None
        geom_ps = _read_ds(f, "/geom/PixelSpacing").reshape(2).astype(np.float64) if "/geom/PixelSpacing" in f else None
        geom_slice_positions = _read_ds(f, "/geom/slice_positions").astype(np.float64) if "/geom/slice_positions" in f else None
        geom_slice_order = _read_ds(f, "/geom/slice_order").astype(np.int32) if "/geom/slice_order" in f else None
        geom_ipps = _read_ds(f, "/geom/IPPs").astype(np.float64) if "/geom/IPPs" in f else None

        vel_path = _first_existing_path(f, ["/data/vel"])
        if vel_path is None:
            raise RuntimeError("mvpack.h5 missing /data/vel")
        vel = _read_ds(f, vel_path).astype(np.float32)

        if vel.ndim != 5:
            raise RuntimeError(
                f"vel must be 5D (Ny,Nx,Nz,3,Nt). Got shape={vel.shape} from {vel_path}"
            )
        if vel.shape[3] != 3:
            if vel.shape[-1] == 3:
                vel = np.transpose(vel, (0, 1, 2, 4, 3)).copy()
            else:
                raise RuntimeError(
                    f"vel axis with 3 components not found. Got shape={vel.shape} from {vel_path}"
                )

        pcmra_path = _first_existing_path(f, ["/data/Vpcmra4D"])
        if pcmra_path is not None:
            pcmra = _read_ds(f, pcmra_path).astype(np.float32)
        else:
            mag_path = _first_existing_path(f, ["/data/mag"])
            if mag_path is None:
                raise RuntimeError("mvpack.h5 missing /data/Vpcmra4D and /data/mag")
            mag = _read_ds(f, mag_path).astype(np.float32)

            if mag.ndim == 5 and mag.shape[3] == 1:
                mag = mag[:, :, :, 0, :]
            if mag.ndim != 4:
                raise RuntimeError(
                    f"mag must be 4D (Ny,Nx,Nz,Nt). Got shape={mag.shape} from {mag_path}"
                )

            Nt = vel.shape[4]
            if mag.shape[3] != Nt:
                raise RuntimeError(f"mag Nt {mag.shape[3]} != vel Nt {Nt}")

            data_group = f["/data"] if "/data" in f else None
            gamma = _read_attr_float(data_group, ["gamma", "ramma"], 0.2)

            v2 = vel[:, :, :, 0, :] ** 2 + vel[:, :, :, 1, :] ** 2 + vel[:, :, :, 2, :] ** 2
            pcmra = (mag * (v2 ** float(gamma))).astype(np.float32)

        if pcmra.ndim != 4:
            if pcmra.ndim == 5 and pcmra.shape[3] == 1:
                pcmra = pcmra[:, :, :, 0, :].astype(np.float32)
            else:
                raise RuntimeError(
                    f"pcmra must be 4D (Ny,Nx,Nz,Nt). Got shape={pcmra.shape} from {pcmra_path}"
                )

        if geom_slice_order is not None:
            print("[mvtracking] slice_order found -> IGNORED")

        print(
            "[mvtracking] geom axis_map="
            f"{geom_axis_map} orgn4={np.array2string(orgn4, precision=4, separator=',')} "
            f"A0={np.array2string(A[:, 0], precision=4, separator=',')}"
        )

        ke = None
        ke_path = _first_existing_path(f, ["/data/ke"])
        if ke_path is not None:
            ke = _read_ds(f, ke_path).astype(np.float32)

        vortmag = None
        vortmag_path = _first_existing_path(f, ["/data/vortmag"])
        if vortmag_path is not None:
            vortmag = _read_ds(f, vortmag_path).astype(np.float32)
        elif "/data/vort" in f:
            vort = _read_ds(f, "/data/vort").astype(np.float32)
            if vort.ndim == 5 and vort.shape[3] == 3:
                vortmag = np.sqrt(
                    vort[:, :, :, 0, :] ** 2 + vort[:, :, :, 1, :] ** 2 + vort[:, :, :, 2, :] ** 2
                ).astype(np.float32)

        if "/cine" not in f:
            raise RuntimeError("mvpack.h5 missing /cine group")

        cine_planes: Dict[str, dict] = {}
        for key in f["/cine"].keys():
            base = f"/cine/{key}"
            img_path = _first_existing_path(f, [base + "/cineI"])
            ipp_path = _first_existing_path(f, [base + "/IPP"])
            iop_path = _first_existing_path(f, [base + "/IOP"])
            ps_path = _first_existing_path(f, [base + "/PixelSpacing"])

            if img_path is None or ipp_path is None or iop_path is None or ps_path is None:
                continue

            img = _read_ds(f, img_path)
            if img.ndim == 3:
                img = _coerce_cine_axes(img, pcmra.shape[0], pcmra.shape[1])
            elif img.ndim == 2:
                img = img[:, :, None]
            img = img.astype(np.float32)

            ipp = _read_ds(f, ipp_path).reshape(3).astype(np.float64)
            iop = _read_ds(f, iop_path).reshape(6).astype(np.float64)
            ps = _read_ds(f, ps_path).reshape(2).astype(np.float64)

            group = f[base]
            cine_axis_map = _read_axis_map_json(group)

            cine_planes[key.lower()] = {
                "img": img,
                "geom": CineGeom(ipp=ipp, iop=iop, ps=ps, axis_map=cine_axis_map),
            }

        if not cine_planes:
            raise RuntimeError("No cine planes found under /cine (expected cineI+IPP+IOP+PixelSpacing)")

        return MVPack(
            cine_planes=cine_planes,
            pcmra=pcmra,
            vel=vel,
            geom=VolGeom(
                orgn4=orgn4,
                A=A,
                axis_map=geom_axis_map,
                iop=geom_iop,
                ipp0=geom_ipp0,
                pixel_spacing=geom_ps,
                slice_positions=geom_slice_positions,
                slice_order=geom_slice_order,
            ),
            ke=ke,
            vortmag=vortmag,
        )
