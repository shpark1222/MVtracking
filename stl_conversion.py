from __future__ import annotations

from typing import Iterable, Tuple

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


def write_ascii_stl(path: str, triangles: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
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
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            for v in verts:
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid mvtrack\n")


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
):
    triangles = plane_roi_to_triangles(
        cine_geom=cine_geom,
        line_xy=line_xy,
        roi_abs_pts=roi_abs_pts,
        npix=npix,
        cine_shape=cine_shape,
        angle_offset_deg=angle_offset_deg,
    )
    write_ascii_stl(out_path, triangles)
