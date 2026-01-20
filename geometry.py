from typing import Dict, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from mvpack_io import CineGeom, VolGeom


def cine_line_to_patient_xyz(line_xy: np.ndarray, cine_geom: CineGeom) -> np.ndarray:
    ipp = cine_geom.ipp.reshape(3)
    iop = cine_geom.iop.reshape(6)
    row_dir = iop[0:3]
    col_dir = iop[3:6]
    ps_row, ps_col = cine_geom.ps.reshape(2)

    cc = line_xy[:, 0]
    rr = line_xy[:, 1]
    P = ipp[None, :] + (cc[:, None] * ps_col) * col_dir[None, :] + (rr[:, None] * ps_row) * row_dir[None, :]
    return P


def cine_plane_normal(cine_geom: CineGeom) -> np.ndarray:
    iop = cine_geom.iop.reshape(6)
    row_dir = iop[0:3]
    col_dir = iop[3:6]
    n = np.cross(row_dir, col_dir)
    nn = np.linalg.norm(n)
    return n / (nn if nn > 0 else 1e-12)


def make_plane_from_cine_line(line_xy: np.ndarray, cine_geom: CineGeom):
    P = cine_line_to_patient_xyz(line_xy, cine_geom)
    c = P.mean(axis=0)

    n_cine = cine_plane_normal(cine_geom)

    u0 = P[1] - P[0]
    u0 = u0 - n_cine * np.dot(u0, n_cine)
    un = np.linalg.norm(u0)
    u = u0 / (un if un > 0 else 1e-12)

    n = np.cross(u, n_cine)
    nn = np.linalg.norm(n)
    n = n / (nn if nn > 0 else 1e-12)

    v = np.cross(n, u)
    vn = np.linalg.norm(v)
    v = v / (vn if vn > 0 else 1e-12)

    return c, u, v, n


def auto_fov_from_line(line_xy: np.ndarray, cine_geom: CineGeom) -> float:
    ps_row, ps_col = cine_geom.ps.reshape(2)
    dx = (line_xy[-1, 0] - line_xy[0, 0]) * ps_col
    dy = (line_xy[-1, 1] - line_xy[0, 1]) * ps_row
    length = float(np.hypot(dx, dy))
    if not np.isfinite(length) or length <= 0:
        length = 40.0
    return float(np.clip(length * 3.0, 150.0, 1000.0))


def reslice_plane_fixedN(
    pcmra3d: np.ndarray,
    vel5d: np.ndarray,
    t: int,
    vol_geom: VolGeom,
    cine_geom: CineGeom,
    line_xy: np.ndarray,
    Npix: int = 192,
    extra_scalars: Optional[Dict[str, np.ndarray]] = None,
):
    c, u, v, n = make_plane_from_cine_line(line_xy, cine_geom)
    fov_half = auto_fov_from_line(line_xy, cine_geom)

    uu = np.linspace(-fov_half, fov_half, Npix)
    vv = np.linspace(-fov_half, fov_half, Npix)
    U, V = np.meshgrid(uu, vv, indexing="xy")
    spmm_eff = float((2.0 * fov_half) / max(Npix - 1, 1))

    XYZ = c.reshape(3, 1) + u.reshape(3, 1) * U.reshape(1, -1) + v.reshape(3, 1) * V.reshape(1, -1)

    A = vol_geom.A
    orgn4 = vol_geom.orgn4.reshape(3, 1)
    abc = np.linalg.solve(A, (XYZ - orgn4))

    colq = (abc[0, :]).reshape(Npix, Npix)
    rowq = (abc[1, :]).reshape(Npix, Npix)
    slcq = (abc[2, :]).reshape(Npix, Npix)

    coords = np.vstack([rowq.ravel(), colq.ravel(), slcq.ravel()])

    Ipcm = map_coordinates(pcmra3d, coords, order=1, mode="constant", cval=0.0).reshape(Npix, Npix)

    vx = map_coordinates(vel5d[:, :, :, 0, t], coords, order=1, mode="constant", cval=0.0).reshape(Npix, Npix)
    vy = map_coordinates(vel5d[:, :, :, 1, t], coords, order=1, mode="constant", cval=0.0).reshape(Npix, Npix)
    vz = map_coordinates(vel5d[:, :, :, 2, t], coords, order=1, mode="constant", cval=0.0).reshape(Npix, Npix)

    Ivelmag = np.sqrt(vx * vx + vy * vy + vz * vz)
    Vn = vx * n[0] + vy * n[1] + vz * n[2]

    extras: Dict[str, np.ndarray] = {}
    if extra_scalars:
        for key, vol in extra_scalars.items():
            extras[key] = map_coordinates(vol, coords, order=1, mode="constant", cval=0.0).reshape(Npix, Npix)

    return Ipcm, Ivelmag, Vn, spmm_eff, extras
