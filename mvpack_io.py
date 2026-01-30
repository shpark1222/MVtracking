import json
import os
import glob
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from scipy.io import loadmat


@dataclass
class CineGeom:
    ipp: np.ndarray
    iop: np.ndarray
    ps: np.ndarray
    edges: Optional[np.ndarray] = None
    axis_map: Optional[Dict[str, np.ndarray]] = None


@dataclass
class VolGeom:
    orgn4: np.ndarray
    A: np.ndarray
    edges: Optional[np.ndarray] = None
    axis_map: Optional[Dict[str, np.ndarray]] = None
    iop: Optional[np.ndarray] = None
    ipp0: Optional[np.ndarray] = None
    ipps: Optional[np.ndarray] = None
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


def load_mrstruct(path: str) -> Tuple[np.ndarray, Dict[str, Optional[np.ndarray]]]:
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    ms = md["mrStruct"]
    edges = getattr(ms, "edges", None)
    vox = getattr(ms, "vox", None)
    meta = {
        "edges": np.array(edges) if edges is not None else None,
        "vox": np.array(vox) if vox is not None else None,
    }
    return np.array(ms.dataAy), meta


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


def _edges_to_A_orgn4(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=np.float64)
    if edges.shape == (3, 4):
        edges = np.vstack([edges, np.array([0.0, 0.0, 0.0, 1.0])])
    if edges.shape != (4, 4):
        raise ValueError(f"Invalid edges shape: {edges.shape}")
    return edges[:3, :3].copy(), edges[:3, 3].copy()


def _build_cine_edges(iop: np.ndarray, ipp: np.ndarray, ps: np.ndarray, slice_step: float) -> np.ndarray:
    # edges: voxel index [x(col), y(row), z(slice), 1] -> patient(mm)
    X = iop[:3] / (np.linalg.norm(iop[:3]) if np.linalg.norm(iop[:3]) > 0 else 1.0)
    Y = iop[3:] / (np.linalg.norm(iop[3:]) if np.linalg.norm(iop[3:]) > 0 else 1.0)
    Z = np.cross(X, Y)
    Z = Z / (np.linalg.norm(Z) if np.linalg.norm(Z) > 0 else 1.0)
    edges = np.eye(4, dtype=np.float64)
    edges[:3, 0] = X * ps[1]
    edges[:3, 1] = Y * ps[0]
    edges[:3, 2] = Z * slice_step
    edges[:3, 3] = ipp
    return edges


def _sanity_check_cine_edges(label: str, cine_geom: CineGeom) -> None:
    edges = cine_geom.edges
    if edges is None:
        return
    edges = np.asarray(edges, dtype=np.float64)
    if edges.shape == (3, 4):
        edges = np.vstack([edges, np.array([0.0, 0.0, 0.0, 1.0])])
    if edges.shape != (4, 4):
        return

    col_vec = edges[:3, 0]
    row_vec = edges[:3, 1]
    row_n = np.linalg.norm(row_vec)
    col_n = np.linalg.norm(col_vec)
    if row_n < 1e-8 or col_n < 1e-8:
        print(f"[mvtracking] cine edges sanity check failed (zero spacing): {label}")
        return

    row_dir = row_vec / row_n
    col_dir = col_vec / col_n
    iop = cine_geom.iop.reshape(6)
    col_ref = iop[:3] / (np.linalg.norm(iop[:3]) if np.linalg.norm(iop[:3]) > 0 else 1.0)
    row_ref = iop[3:] / (np.linalg.norm(iop[3:]) if np.linalg.norm(iop[3:]) > 0 else 1.0)

    dot_row = float(np.dot(row_dir, row_ref))
    dot_col = float(np.dot(col_dir, col_ref))
    if dot_row < 0.5 or dot_col < 0.5:
        print(
            "[mvtracking] cine edges sanity warning "
            f"({label}) dot_row={dot_row:.3f} dot_col={dot_col:.3f}"
        )


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
        geom_edges = _read_ds(f, "/geom/edges").astype(np.float64) if "/geom/edges" in f else None
        if geom_edges is not None:
            A, orgn4 = _edges_to_A_orgn4(geom_edges)
        else:
            if "/geom/orgn4" not in f or "/geom/A" not in f:
                raise RuntimeError("mvpack.h5 missing /geom/orgn4 or /geom/A (or /geom/edges)")
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
            print("[mvtracking] slice_order found -> stored for slice mapping")

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
                pass
            elif img.ndim == 2:
                img = img[:, :, None]
            img = img.astype(np.float32)

            ipp = _read_ds(f, ipp_path).reshape(3).astype(np.float64)
            iop = _read_ds(f, iop_path).reshape(6).astype(np.float64)
            ps = _read_ds(f, ps_path).reshape(2).astype(np.float64)
            edges = _read_ds(f, base + "/edges").astype(np.float64) if base + "/edges" in f else None
            if edges is None:
                edges = _build_cine_edges(iop, ipp, ps, 1.0)
                print(f"[mvtracking] cine edges missing for {key}; using DICOM fallback.")

            group = f[base]
            cine_axis_map = _read_axis_map_json(group)

            cine_planes[key.lower()] = {
                "img": img,
                "geom": CineGeom(ipp=ipp, iop=iop, ps=ps, edges=edges, axis_map=cine_axis_map),
            }
            _sanity_check_cine_edges(key.lower(), cine_planes[key.lower()]["geom"])

        if not cine_planes:
            raise RuntimeError("No cine planes found under /cine (expected cineI+IPP+IOP+PixelSpacing)")

        return MVPack(
            cine_planes=cine_planes,
            pcmra=pcmra,
            vel=vel,
            geom=VolGeom(
                orgn4=orgn4,
                A=A,
                edges=geom_edges,
                axis_map=geom_axis_map,
                iop=geom_iop,
                ipp0=geom_ipp0,
                ipps=geom_ipps,
                pixel_spacing=geom_ps,
                slice_positions=geom_slice_positions,
                slice_order=geom_slice_order,
            ),
            ke=ke,
            vortmag=vortmag,
        )
