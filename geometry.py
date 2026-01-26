from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from mvpack_io import CineGeom, VolGeom


def apply_axis_transform(
    volume: np.ndarray,
    order: str = "XYZ",
    flips: Optional[Tuple[bool, bool, bool]] = None,
) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim < 3:
        return arr
    if order is None:
        order = "XYZ"
    order = str(order).upper()
    if len(order) != 3 or set(order) != {"X", "Y", "Z"}:
        raise ValueError(f"Invalid axis order '{order}'. Expected permutation of XYZ.")
    axis_map = {"X": 0, "Y": 1, "Z": 2}
    perm = [axis_map[c] for c in order]
    if perm != [0, 1, 2]:
        perm = perm + list(range(3, arr.ndim))
        arr = np.transpose(arr, perm)
    if flips is None:
        flips = (False, False, False)
    if len(flips) < 3:
        flips = tuple(flips) + (False,) * (3 - len(flips))
    for axis, do_flip in enumerate(flips[:3]):
        if do_flip:
            arr = np.flip(arr, axis=axis)
    return arr


def transform_vector_components(
    vectors: np.ndarray,
    order: str = "XYZ",
    flips: Optional[Tuple[bool, bool, bool]] = None,
) -> np.ndarray:
    arr = np.asarray(vectors)
    if arr.ndim < 4:
        return arr
    order = str(order).upper() if order is not None else "XYZ"
    if len(order) != 3 or set(order) != {"X", "Y", "Z"}:
        raise ValueError(f"Invalid axis order '{order}'. Expected permutation of XYZ.")
    axis_map = {"X": 0, "Y": 1, "Z": 2}
    perm = [axis_map[c] for c in order]
    comp_axis = 3
    arr = np.take(arr, perm, axis=comp_axis)
    if flips is None:
        flips = (False, False, False)
    if len(flips) < 3:
        flips = tuple(flips) + (False,) * (3 - len(flips))
    for axis, do_flip in enumerate(flips[:3]):
        if do_flip:
            if arr.ndim == 4:
                arr[..., axis] *= -1.0
            else:
                arr[..., axis, :] *= -1.0
    return arr


def transform_points_axis_order(
    points_xyz: np.ndarray,
    order: str = "XYZ",
    flips: Optional[Tuple[bool, bool, bool]] = None,
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError("points_xyz must have shape (..., 3)")
    order = str(order).upper() if order is not None else "XYZ"
    if len(order) != 3 or set(order) != {"X", "Y", "Z"}:
        raise ValueError(f"Invalid axis order '{order}'. Expected permutation of XYZ.")
    axis_map = {"X": 0, "Y": 1, "Z": 2}
    perm = [axis_map[c] for c in order]
    pts = pts[..., perm]
    if flips is None:
        flips = (False, False, False)
    if len(flips) < 3:
        flips = tuple(flips) + (False,) * (3 - len(flips))
    sign = np.array([(-1.0 if flips[i] else 1.0) for i in range(3)], dtype=np.float64)
    return pts * sign


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


def _cine_geom_components(cine_geom: CineGeom) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    edges = cine_geom.edges
    if edges is not None:
        edges = np.asarray(edges, dtype=np.float64)
        if edges.shape == (3, 4):
            edges = np.vstack([edges, np.array([0.0, 0.0, 0.0, 1.0])])
        # edges: voxel index [x(col), y(row), z(slice), 1] -> patient(mm)
        col_vec = edges[:3, 0]
        row_vec = edges[:3, 1]
        ps_col = float(np.linalg.norm(col_vec))
        ps_row = float(np.linalg.norm(row_vec))
        col_dir = col_vec / (ps_col if ps_col > 0 else 1e-12)
        row_dir = row_vec / (ps_row if ps_row > 0 else 1e-12)
        ipp = edges[:3, 3].astype(np.float64)
        return ipp, row_dir, col_dir, ps_row, ps_col

    ipp = cine_geom.ipp.reshape(3)
    iop = cine_geom.iop.reshape(6)
    col_dir = _normalize(iop[0:3])
    row_dir = _normalize(iop[3:6])
    ps_row, ps_col = cine_geom.ps.reshape(2)
    return ipp, row_dir, col_dir, float(ps_row), float(ps_col)


def cine_display_axes(cine_geom: CineGeom) -> Dict[str, np.ndarray]:
    _, row_dir, col_dir, _, _ = _cine_geom_components(cine_geom)
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
    ipp, row_dir, col_dir, ps_row, ps_col = _cine_geom_components(cine_geom)
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
    ipp, row_dir, col_dir, ps_row, ps_col = _cine_geom_components(cine_geom)
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
    ipp, row_dir, col_dir, ps_row, ps_col = _cine_geom_components(cine_geom)
    geom = DICOMGeometry2D(ipp=ipp, row_cos=row_dir, col_cos=col_dir, ps_row=ps_row, ps_col=ps_col)
    rr = line_xy[:, 1]
    cc = line_xy[:, 0]
    return geom.pixel_to_patient(rr, cc)


def cine_plane_normal(cine_geom: CineGeom) -> np.ndarray:
    _, row_dir, col_dir, _, _ = _cine_geom_components(cine_geom)
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
    _, _, _, ps_row, ps_col = _cine_geom_components(cine_geom)
    dx = (line_xy[-1, 0] - line_xy[0, 0]) * ps_col
    dy = (line_xy[-1, 1] - line_xy[0, 1]) * ps_row
    length = float(np.hypot(dx, dy))
    if not np.isfinite(length) or length <= 0:
        length = 40.0
    return float(np.clip(length * 3.2, 150.0, 1000.0))


def reslice_cine_to_pcmra_grid(
    pcmra_edges: np.ndarray,
    pcmra_shape: Tuple[int, ...],
    cine_img: np.ndarray,
    cine_edges: np.ndarray,
    z_tolerance: float = 0.5,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Reslice cine into the PCMRA grid using edges only.
    edges: voxel index [x(col), y(row), z(slice), 1] -> patient(mm)
    """
    pcmra_edges = np.asarray(pcmra_edges, dtype=np.float64)
    cine_edges = np.asarray(cine_edges, dtype=np.float64)
    if pcmra_edges.shape == (3, 4):
        pcmra_edges = np.vstack([pcmra_edges, np.array([0.0, 0.0, 0.0, 1.0])])
    if cine_edges.shape == (3, 4):
        cine_edges = np.vstack([cine_edges, np.array([0.0, 0.0, 0.0, 1.0])])

    if pcmra_edges.shape != (4, 4) or cine_edges.shape != (4, 4):
        raise RuntimeError("pcmra_edges and cine_edges must be 4x4 (or 3x4)")

    if cine_img.ndim == 3:
        cine_img = cine_img.mean(axis=2)

    Ny = int(pcmra_shape[0])
    Nx = int(pcmra_shape[1])
    Nz = int(pcmra_shape[2]) if len(pcmra_shape) > 2 else 1

    idx = np.indices((Ny, Nx, Nz), dtype=np.float64)
    row = idx[0].reshape(1, -1)
    col = idx[1].reshape(1, -1)
    slc = idx[2].reshape(1, -1)
    hom = np.vstack([col, row, slc, np.ones_like(col)])

    P = pcmra_edges @ hom
    cine_vox = np.linalg.inv(cine_edges) @ P
    x_cine = cine_vox[0, :]
    y_cine = cine_vox[1, :]
    z_cine = cine_vox[2, :]

    coords = np.vstack([y_cine, x_cine])
    sampled = map_coordinates(cine_img, coords, order=1, mode="constant", cval=fill_value)
    if z_tolerance is not None:
        mask = np.abs(z_cine) <= float(z_tolerance)
        sampled = np.where(mask, sampled, fill_value)

    out = sampled.reshape(Ny, Nx, Nz)
    return out[:, :, 0] if Nz == 1 else out


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

    edges = vol_geom.edges
    if edges is not None:
        edges = np.asarray(edges, dtype=np.float64)
        if edges.shape == (3, 4):
            edges = np.vstack([edges, np.array([0.0, 0.0, 0.0, 1.0])])
        if edges.shape != (4, 4):
            raise RuntimeError(f"Invalid volume edges shape: {edges.shape}")
        hom = np.vstack([XYZ, np.ones((1, XYZ.shape[1]), dtype=np.float64)])
        vox = np.linalg.inv(edges) @ hom
        colq = vox[0, :].reshape(Npix, Npix)
        rowq = vox[1, :].reshape(Npix, Npix)
        slcq = vox[2, :].reshape(Npix, Npix)
    elif vol_geom.ipps is not None and vol_geom.slice_positions is not None:
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
        slc_dir = _unit(np.cross(col_dir, row_dir))
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
