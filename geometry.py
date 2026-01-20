from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from mvpack_io import CineGeom, VolGeom


@dataclass
class DICOMGeometry2D:
    ipp: np.ndarray
    row_cos: np.ndarray
    col_cos: np.ndarray
    ps_row: float
    ps_col: float

    def pixel_to_patient(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        i = np.asarray(i, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        return (
            self.ipp[None, :]
            + (i[:, None] * self.ps_row) * self.row_cos[None, :]
            + (j[:, None] * self.ps_col) * self.col_cos[None, :]
        )

    def patient_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        A = np.column_stack([self.row_cos * self.ps_row, self.col_cos * self.ps_col])
        ij = np.linalg.lstsq(A, (xyz - self.ipp).T, rcond=None)[0].T
        return ij


@dataclass
class DICOMGeometry3D:
    org: np.ndarray
    row_cos: np.ndarray
    col_cos: np.ndarray
    slice_cos: np.ndarray
    spacing_row: float
    spacing_col: float
    spacing_slice: float

    def voxel_to_patient(self, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        i = np.asarray(i, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        return (
            self.org[None, :]
            + (i[:, None] * self.spacing_row) * self.row_cos[None, :]
            + (j[:, None] * self.spacing_col) * self.col_cos[None, :]
            + (k[:, None] * self.spacing_slice) * self.slice_cos[None, :]
        )

    def patient_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        A = np.column_stack(
            [
                self.row_cos * self.spacing_row,
                self.col_cos * self.spacing_col,
                self.slice_cos * self.spacing_slice,
            ]
        )
        ijk = np.linalg.solve(A, (xyz - self.org).T).T
        return ijk


def cine_line_to_patient_xyz(line_xy: np.ndarray, cine_geom: CineGeom) -> np.ndarray:
    ipp = cine_geom.ipp.reshape(3)
    iop = cine_geom.iop.reshape(6)
    col_dir = iop[0:3]
    row_dir = iop[3:6]
    ps_row, ps_col = cine_geom.ps.reshape(2)
    geom = DICOMGeometry2D(ipp=ipp, row_cos=row_dir, col_cos=col_dir, ps_row=ps_row, ps_col=ps_col)
    rr = line_xy[:, 1]
    cc = line_xy[:, 0]
    return geom.pixel_to_patient(rr, cc)


def cine_plane_normal(cine_geom: CineGeom) -> np.ndarray:
    iop = cine_geom.iop.reshape(6)
    col_dir = iop[0:3]
    row_dir = iop[3:6]
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
