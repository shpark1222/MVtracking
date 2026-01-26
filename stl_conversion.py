from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


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


def _polygon_normal(contour_pts_xyz: np.ndarray) -> Optional[np.ndarray]:
    contour = np.asarray(contour_pts_xyz, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 3:
        return None
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    if contour.shape[0] < 3:
        return None
    normal = np.zeros(3, dtype=np.float64)
    for idx in range(contour.shape[0]):
        p0 = contour[idx]
        p1 = contour[(idx + 1) % contour.shape[0]]
        normal[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
        normal[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
        normal[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    return normal / norm


def _extrude_contour_triangles(
    contour_pts_xyz: np.ndarray,
    thickness_mm: float,
) -> Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    contour = np.asarray(contour_pts_xyz, dtype=np.float64)
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    if contour.shape[0] < 3:
        raise ValueError("contour_pts_xyz must include at least 3 unique points")
    normal = _polygon_normal(contour)
    if normal is None:
        return _triangulate_contour_fan(contour)
    half = 0.5 * thickness_mm
    top = contour + normal * half
    bottom = contour - normal * half

    triangles = []
    triangles.extend(_triangulate_contour_fan(top))
    triangles.extend(_triangulate_contour_fan(bottom[::-1]))

    for idx in range(contour.shape[0]):
        nxt = (idx + 1) % contour.shape[0]
        top0 = top[idx]
        top1 = top[nxt]
        bot0 = bottom[idx]
        bot1 = bottom[nxt]
        triangles.append((top0, top1, bot1))
        triangles.append((top0, bot1, bot0))
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


def write_stl_from_patient_contour_extruded(
    out_path: str,
    contour_pts_xyz: Optional[np.ndarray],
    thickness_mm: float,
    output_space: str = "LPS",
):
    """Write an STL by extruding a patient-space contour along its normal."""
    if contour_pts_xyz is None:
        raise ValueError("contour_pts_xyz is required for extruded STL output")
    if not np.isfinite(thickness_mm) or thickness_mm <= 0.0:
        write_stl_from_patient_contour(out_path, contour_pts_xyz, output_space=output_space)
        return
    triangles = _extrude_contour_triangles(np.asarray(contour_pts_xyz, dtype=np.float64), thickness_mm)
    write_ascii_stl(out_path, triangles, output_space=output_space)
