from __future__ import annotations

from typing import Iterable, Tuple, Optional, Sequence

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

from geometry import auto_fov_from_line, make_plane_from_cine_line
from mvpack_io import CineGeom, VolGeom
from roi_utils import polygon_mask


def _plane_voxel_coords(
    vol_geom: VolGeom,
    cine_geom: CineGeom,
    line_xy: np.ndarray,
    npix: int,
    cine_shape: Tuple[int, int] | None = None,
    angle_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c, u, v, _ = make_plane_from_cine_line(
        line_xy,
        cine_geom,
        cine_shape=cine_shape,
        angle_offset_deg=angle_offset_deg,
    )
    fov_half = auto_fov_from_line(line_xy, cine_geom)

    uu = np.linspace(-fov_half, fov_half, npix)
    vv = np.linspace(-fov_half, fov_half, npix)
    U, V = np.meshgrid(uu, vv, indexing="xy")

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

        colq = uv[0, :].reshape(npix, npix)
        rowq = uv[1, :].reshape(npix, npix)
        slcq = k.reshape(npix, npix)
    else:
        A = vol_geom.A
        orgn4 = vol_geom.orgn4.reshape(3, 1)
        abc = np.linalg.solve(A, (XYZ - orgn4))

        colq = abc[0, :].reshape(npix, npix)
        rowq = abc[1, :].reshape(npix, npix)
        slcq = abc[2, :].reshape(npix, npix)

    return rowq, colq, slcq


def plane_roi_to_triangles(
    cine_geom: CineGeom,
    line_xy: np.ndarray,
    roi_abs_pts: np.ndarray,
    npix: int = 192,
    cine_shape: Tuple[int, int] | None = None,
    angle_offset_deg: float = 0.0,
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    c, u, v, _ = make_plane_from_cine_line(
        line_xy,
        cine_geom,
        cine_shape=cine_shape,
        angle_offset_deg=angle_offset_deg,
    )
    fov_half = auto_fov_from_line(line_xy, cine_geom)

    uu = np.linspace(-fov_half, fov_half, npix)
    vv = np.linspace(-fov_half, fov_half, npix)

    mask2d = polygon_mask((npix, npix), roi_abs_pts)
    if not np.any(mask2d):
        return []

    mask2d = binary_closing(mask2d, structure=np.ones((3, 3)), iterations=2)
    mask2d = binary_fill_holes(mask2d)

    triangles = []
    for r in range(npix - 1):
        for c_idx in range(npix - 1):
            if not (mask2d[r, c_idx] and mask2d[r + 1, c_idx] and mask2d[r, c_idx + 1] and mask2d[r + 1, c_idx + 1]):
                continue
            uv00 = np.array([uu[c_idx], vv[r]])
            uv10 = np.array([uu[c_idx + 1], vv[r]])
            uv01 = np.array([uu[c_idx], vv[r + 1]])
            uv11 = np.array([uu[c_idx + 1], vv[r + 1]])

            v00 = c + uv00[0] * u + uv00[1] * v
            v10 = c + uv10[0] * u + uv10[1] * v
            v01 = c + uv01[0] * u + uv01[1] * v
            v11 = c + uv11[0] * u + uv11[1] * v

            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    return triangles


def plane_roi_to_mask(
    vol_geom: VolGeom,
    cine_geom: CineGeom,
    line_xy: np.ndarray,
    roi_abs_pts: np.ndarray,
    vol_shape: Tuple[int, int, int],
    npix: int = 192,
    cine_shape: Tuple[int, int] | None = None,
    angle_offset_deg: float = 0.0,
) -> np.ndarray:
    rowq, colq, slcq = _plane_voxel_coords(
        vol_geom,
        cine_geom,
        line_xy,
        npix,
        cine_shape=cine_shape,
        angle_offset_deg=angle_offset_deg,
    )
    mask2d = polygon_mask((npix, npix), roi_abs_pts)
    if not np.any(mask2d):
        return np.zeros(vol_shape, dtype=np.uint8)

    idx = np.where(mask2d)
    rows = np.rint(rowq[idx]).astype(np.int64)
    cols = np.rint(colq[idx]).astype(np.int64)
    slcs = np.rint(slcq[idx]).astype(np.int64)

    ny, nx, nz = vol_shape
    valid = (
        (rows >= 0) & (rows < ny) &
        (cols >= 0) & (cols < nx) &
        (slcs >= 0) & (slcs < nz)
    )
    rows = rows[valid]
    cols = cols[valid]
    slcs = slcs[valid]

    vol = np.zeros(vol_shape, dtype=np.uint8)
    vol[rows, cols, slcs] = 1
    return vol


def _lps_to_ras(xyz: np.ndarray) -> np.ndarray:
    return xyz * np.array([-1.0, -1.0, 1.0])


def _apply_output_space(xyz: np.ndarray, output_space: str) -> np.ndarray:
    space = output_space.upper()
    if space == "LPS":
        return xyz
    if space == "RAS":
        return _lps_to_ras(xyz)
    raise ValueError(f"Unsupported output_space={output_space!r}. Expected 'LPS' or 'RAS'.")


def write_ascii_stl(
    path: str,
    triangles: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_space: str = "RAS",
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("solid mvtrack\n")
        for v0, v1, v2 in triangles:
            verts = np.vstack([v0, v1, v2])
            n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
            else:
                n = np.array([0.0, 0.0, 0.0])
            verts = _apply_output_space(verts, output_space)
            n = _apply_output_space(n, output_space)
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            for v in verts:
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid mvtrack\n")


def _normalize_triangles(
    triangles: Optional[Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> Optional[Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    if triangles is None:
        return None
    if isinstance(triangles, np.ndarray):
        arr = np.asarray(triangles, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] != (3, 3):
            raise ValueError("triangles array must have shape (N, 3, 3)")
        return [(arr[i, 0], arr[i, 1], arr[i, 2]) for i in range(arr.shape[0])]
    return triangles


def _triangulate_contour_fan(contour_pts_xyz: np.ndarray) -> Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    contour = np.asarray(contour_pts_xyz, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 3:
        raise ValueError("contour_pts_xyz must be (N, 3) with N >= 3")
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    if contour.shape[0] < 3:
        raise ValueError("contour_pts_xyz must include at least 3 unique points")
    center = contour.mean(axis=0)
    triangles = []
    for idx in range(contour.shape[0]):
        p0 = contour[idx]
        p1 = contour[(idx + 1) % contour.shape[0]]
        triangles.append((center, p0, p1))
    return triangles


def write_stl_from_patient_contour(
    out_path: str,
    contour_pts_xyz: Optional[np.ndarray],
    output_space: str = "LPS",
    triangles: Optional[Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
):
    """Write an STL from patient-space contour points (assumed LPS) or precomputed triangles."""
    triangles_norm = _normalize_triangles(triangles)
    if triangles_norm is None:
        if contour_pts_xyz is None:
            raise ValueError("contour_pts_xyz is required when triangles are not provided")
        triangles_norm = _triangulate_contour_fan(np.asarray(contour_pts_xyz, dtype=np.float64))
    write_ascii_stl(out_path, triangles_norm, output_space=output_space)


def convert_plane_to_stl(
    out_path: str,
    vol_geom: VolGeom,
    cine_geom: CineGeom,
    line_xy: np.ndarray,
    roi_abs_pts: np.ndarray,
    vol_shape: Tuple[int, int, int],
    npix: int = 192,
    cine_shape: Tuple[int, int] | None = None,
    angle_offset_deg: float = 0.0,
    output_space: str = "RAS",
):
    triangles = plane_roi_to_triangles(
        cine_geom=cine_geom,
        line_xy=line_xy,
        roi_abs_pts=roi_abs_pts,
        npix=npix,
        cine_shape=cine_shape,
        angle_offset_deg=angle_offset_deg,
    )
    write_ascii_stl(out_path, triangles, output_space=output_space)
