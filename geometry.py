from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(vec)
    return vec / (n if n > 0 else 1e-12)


def cine_display_axes(cine_geom: CineGeom) -> Dict[str, np.ndarray]:
    iop = cine_geom.iop.reshape(6)
    col_dir = _normalize(iop[0:3])
    row_dir = _normalize(iop[3:6])
    n_cine = _normalize(np.cross(col_dir, row_dir))

    eP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    eS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    x0 = eP - np.dot(eP, n_cine) * n_cine
    y0 = eS - np.dot(eS, n_cine) * n_cine
    if np.linalg.norm(x0) < 1e-6:
        x0 = col_dir - np.dot(col_dir, n_cine) * n_cine
    if np.linalg.norm(y0) < 1e-6:
        y0 = row_dir - np.dot(row_dir, n_cine) * n_cine

    x_disp = _normalize(x0)
    y_tmp = y0 - np.dot(y0, x_disp) * x_disp
    if np.linalg.norm(y_tmp) < 1e-6:
        y_tmp = np.cross(n_cine, x_disp)
    y_disp = _normalize(y_tmp)

    if np.dot(np.cross(x_disp, y_disp), n_cine) < 0:
        y_disp = -y_disp
    if np.dot(y_disp, eS) < 0:
        y_disp = -y_disp

    return {"x_disp": x_disp, "y_disp": y_disp, "n_cine": n_cine, "col_dir": col_dir, "row_dir": row_dir}


def cine_display_pixel_to_patient(
    line_xy: np.ndarray,
    cine_geom: CineGeom,
    cine_shape: Tuple[int, int],
) -> np.ndarray:
    H, W = cine_shape
    ps_row, ps_col = cine_geom.ps.reshape(2)
    ipp = cine_geom.ipp.reshape(3)
    axes = cine_display_axes(cine_geom)
    x_disp = axes["x_disp"]
    y_disp = axes["y_disp"]
    col_dir = axes["col_dir"]
    row_dir = axes["row_dir"]

    Pc = (
        ipp
        + ((W - 1) / 2.0) * col_dir * ps_col
        + ((H - 1) / 2.0) * row_dir * ps_row
    )

    u = (line_xy[:, 0] - (W - 1) / 2.0) * ps_col
    v = (line_xy[:, 1] - (H - 1) / 2.0) * ps_row
    v = -v
    return Pc[None, :] + u[:, None] * x_disp[None, :] + v[:, None] * y_disp[None, :]


def cine_display_mapping(
    cine_geom: CineGeom,
    cine_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = cine_shape
    ps_row, ps_col = cine_geom.ps.reshape(2)
    ipp = cine_geom.ipp.reshape(3)
    axes = cine_display_axes(cine_geom)
    x_disp = axes["x_disp"]
    y_disp = axes["y_disp"]
    col_dir = axes["col_dir"]
    row_dir = axes["row_dir"]

    Pc = (
        ipp
        + ((W - 1) / 2.0) * col_dir * ps_col
        + ((H - 1) / 2.0) * row_dir * ps_row
    )

    u = (np.arange(W, dtype=np.float64) - (W - 1) / 2.0) * ps_col
    v = (np.arange(H, dtype=np.float64) - (H - 1) / 2.0) * ps_row
    v = -v
    U, V = np.meshgrid(u, v, indexing="xy")
    Pdisp = Pc.reshape(1, 1, 3) + U[:, :, None] * x_disp[None, None, :] + V[:, :, None] * y_disp[None, None, :]

    geom = DICOMGeometry2D(ipp=ipp, row_cos=row_dir, col_cos=col_dir, ps_row=ps_row, ps_col=ps_col)
    ij = geom.patient_to_pixel(Pdisp.reshape(-1, 3))
    rowq = ij[:, 0].reshape(H, W)
    colq = ij[:, 1].reshape(H, W)
    return rowq, colq


def cine_line_to_patient_xyz(
    line_xy: np.ndarray,
    cine_geom: CineGeom,
    cine_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    if cine_shape is not None:
        return cine_display_pixel_to_patient(line_xy, cine_geom, cine_shape)
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
    n = np.cross(col_dir, row_dir)
    nn = np.linalg.norm(n)
    return n / (nn if nn > 0 else 1e-12)


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    vec = np.asarray(vec, dtype=np.float64)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def make_plane_from_cine_line(
    line_xy: np.ndarray,
    cine_geom: CineGeom,
    cine_shape: Optional[Tuple[int, int]] = None,
    angle_offset_deg: float = 0.0,
):
    P = cine_line_to_patient_xyz(line_xy, cine_geom, cine_shape=cine_shape)
    c = P.mean(axis=0)

    n_cine = cine_plane_normal(cine_geom)

    u0 = P[1] - P[0]
    u0 = u0 - n_cine * np.dot(u0, n_cine)
    un = np.linalg.norm(u0)
    u = u0 / (un if un > 0 else 1e-12)
    if angle_offset_deg:
        u = _rotate_vector(u, n_cine, np.deg2rad(angle_offset_deg))

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
    cine_shape: Optional[Tuple[int, int]] = None,
    angle_offset_deg: float = 0.0,
    Npix: int = 192,
    extra_scalars: Optional[Dict[str, np.ndarray]] = None,
):
    c, u, v, n = make_plane_from_cine_line(
        line_xy,
        cine_geom,
        cine_shape=cine_shape,
        angle_offset_deg=angle_offset_deg,
    )
    fov_half = auto_fov_from_line(line_xy, cine_geom)

    uu = np.linspace(-fov_half, fov_half, Npix)
    vv = np.linspace(-fov_half, fov_half, Npix)
    U, V = np.meshgrid(uu, vv, indexing="xy")
    spmm_eff = float((2.0 * fov_half) / max(Npix - 1, 1))

    XYZ = c.reshape(3, 1) + u.reshape(3, 1) * U.reshape(1, -1) + v.reshape(3, 1) * V.reshape(1, -1)

    if vol_geom.ipps is not None and vol_geom.slice_positions is not None:
        print("[mvtracking] using per-slice IPP/IOP mapping")
        iop = vol_geom.iop
        if iop is None:
            raise RuntimeError("vol_geom.iop is required for per-slice mapping")
        ps = vol_geom.pixel_spacing
        if ps is None:
            raise RuntimeError("vol_geom.pixel_spacing is required for per-slice mapping")
        ipps = vol_geom.ipps
        slice_positions = vol_geom.slice_positions

        def _unit(vec: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(vec)
            if n < 1e-8:
                return vec * 0.0
            return vec / n

        col_dir = _unit(iop[0:3])
        row_dir = _unit(iop[3:6])
        slc_dir = _unit(np.cross(row_dir, col_dir))
        if ipps.shape[0] >= 2:
            d = ipps[-1] - ipps[0]
            if np.dot(slc_dir, d) < 0:
                slc_dir = -slc_dir

        p = XYZ.T @ slc_dir
        idx = np.searchsorted(slice_positions, p)
        idx0 = np.clip(idx, 0, len(slice_positions) - 1)
        idx1 = np.clip(idx - 1, 0, len(slice_positions) - 1)
        use0 = np.abs(p - slice_positions[idx0]) <= np.abs(p - slice_positions[idx1])
        k = np.where(use0, idx0, idx1)

        ipp_k = ipps[k]
        B = np.column_stack([col_dir * ps[1], row_dir * ps[0]])
        uv = np.linalg.pinv(B) @ (XYZ - ipp_k.T)

        colq = uv[0, :].reshape(Npix, Npix)
        rowq = uv[1, :].reshape(Npix, Npix)
        slcq = k.reshape(Npix, Npix)
    else:
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
