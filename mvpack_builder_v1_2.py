import os
import sys
import json
import numpy as np
import h5py

from scipy.io import loadmat
from scipy.ndimage import median_filter
from PySide6 import QtWidgets
import pydicom


# ============================
# helpers
# ============================

def _unit(v):
    v = np.asarray(v, float).reshape(3)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1e-12)


def infer_axis_map_from_iop_ipp(iop6, ipps=None):
    # geometry.py convention:
    # iop6[:3] = col_dir (+j), iop6[3:] = row_dir (+i)
    col = _unit(iop6[:3])
    row = _unit(iop6[3:])
    slc = _unit(np.cross(col, row))

    if ipps is not None and len(ipps) >= 2:
        d = ipps[-1] - ipps[0]
        if np.dot(slc, d) < 0:
            slc = -slc

    return {
        "Rows": row.tolist(),
        "Columns": col.tolist(),
        "Slices": slc.tolist(),
    }


def cine_edges_from_dicom(ds0) -> np.ndarray:
    # edges: voxel index [x(col), y(row), z(slice), 1] -> patient(mm)
    iop = np.array(ds0.ImageOrientationPatient, float)  # [X..., Y...]
    ipp = np.array(ds0.ImagePositionPatient, float)
    ps = np.array(ds0.PixelSpacing, float)              # [rowSpacing, colSpacing]
    row_spacing = float(ps[0])
    col_spacing = float(ps[1])

    X = _unit(iop[:3])  # +j (col)
    Y = _unit(iop[3:])  # +i (row)
    Z = _unit(np.cross(X, Y))

    slice_step = float(
        getattr(ds0, "SpacingBetweenSlices", None)
        or getattr(ds0, "SliceThickness", None)
        or 1.0
    )

    edges = np.eye(4, dtype=np.float64)
    edges[:3, 0] = X * col_spacing
    edges[:3, 1] = Y * row_spacing
    edges[:3, 2] = Z * slice_step
    edges[:3, 3] = ipp
    return edges


def volume_edges_from_dicom_series(infos) -> np.ndarray:
    """
    4D DICOM series로부터 volume edges를 만든다.
    edges: voxel index [x(col), y(row), z(slice), 1] -> patient(mm)
    """
    ds0 = infos[0][2]
    iop = np.array(ds0.ImageOrientationPatient, float)  # [col_dir, row_dir]
    ipp0 = np.array(ds0.ImagePositionPatient, float)
    ps = np.array(ds0.PixelSpacing, float)

    col_dir = _unit(iop[:3])    # +j
    row_dir = _unit(iop[3:])    # +i

    ipps = np.array([np.array(ds.ImagePositionPatient, float) for _, _, ds in infos], dtype=np.float64)

    # slice direction + spacing
    slc_dir = _unit(np.cross(col_dir, row_dir))

    if len(ipps) >= 2:
        d = ipps[-1] - ipps[0]
        if np.dot(slc_dir, d) < 0:
            slc_dir = -slc_dir

    proj = ipps @ slc_dir
    diffs = np.diff(np.sort(proj))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[np.abs(diffs) > 1e-6]
    dz = float(np.median(diffs)) if diffs.size else float(ps[0])
    dz = max(dz, 1e-3)

    edges = np.eye(4, dtype=np.float64)
    edges[:3, 0] = col_dir * ps[1]
    edges[:3, 1] = row_dir * ps[0]
    edges[:3, 2] = slc_dir * dz
    edges[:3, 3] = ipp0
    return edges


def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return x


# ============================
# IO
# ============================

def load_mrstruct(path):
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    ms = md["mrStruct"]
    edges = getattr(ms, "edges", None)
    vox = getattr(ms, "vox", None)
    meta = {
        "edges": np.array(edges) if edges is not None else None,
        "vox": np.array(vox) if vox is not None else None,
    }
    return np.array(ms.dataAy), meta


def read_dicom_sorted(folder):
    infos = []
    for r, _, fs in os.walk(folder):
        for f in fs:
            if f.startswith("."):
                continue
            p = os.path.join(r, f)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                inst = int(getattr(ds, "InstanceNumber", 1e9))
                infos.append((inst, p, ds))
            except Exception:
                pass
    infos.sort(key=lambda x: x[0])
    return infos


def estimate_volume_geom(folder):
    infos = read_dicom_sorted(folder)
    if not infos:
        raise RuntimeError(f"No DICOM found under: {folder}")

    ds0 = infos[0][2]
    iop = np.array(ds0.ImageOrientationPatient, float)
    ps = np.array(ds0.PixelSpacing, float)

    ipps = np.array([np.array(ds.ImagePositionPatient, float) for _, _, ds in infos], dtype=np.float64)

    # edges: voxel index [x(col), y(row), z(slice), 1] -> patient(mm)
    edges = volume_edges_from_dicom_series(infos)

    # A/orgn4는 edges에서 파생
    A = edges[:3, :3].copy()
    orgn4 = edges[:3, 3].copy()

    # slice 관련도 저장 (axis_map은 DICOM 기준이지만 여기선 참고용)
    col_dir = _unit(iop[:3])
    row_dir = _unit(iop[3:])
    slc_dir = _unit(np.cross(col_dir, row_dir))
    if len(ipps) >= 2:
        d = ipps[-1] - ipps[0]
        if np.dot(slc_dir, d) < 0:
            slc_dir = -slc_dir
    proj = ipps @ slc_dir
    diffs = np.diff(np.sort(proj))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[np.abs(diffs) > 1e-6]
    dz = float(np.median(diffs)) if diffs.size else float(ps[0])
    dz = max(dz, 1e-3)

    if np.linalg.matrix_rank(A) < 3:
        raise RuntimeError("Invalid volume geometry: A is singular (check IOP/IPP/spacing)")

    return {
        "orgn4": orgn4,
        "A": A,
        "edges": edges,
        "PixelSpacing": ps,
        "sliceStep": np.array([dz]),
        "IOP": iop,
        "IPPs": ipps,
        "slice_positions": proj,
        "slice_order": np.argsort(proj),
        "axis_map": infer_axis_map_from_iop_ipp(iop, ipps),
    }


def read_cine(folder, log_fn=None):
    infos = read_dicom_sorted(folder)
    if not infos:
        raise RuntimeError(f"No cine DICOM found under: {folder}")

    frames = []
    for _, p, ds in infos:
        ds_full = pydicom.dcmread(p, force=True)
        arr = ds_full.pixel_array
        if arr.ndim == 2:
            frames.append(arr)
        elif arr.ndim == 3:
            frames.extend(arr)
        else:
            raise RuntimeError(f"Unsupported cine pixel_array shape: {arr.shape}")

    cine = np.stack(frames, axis=0)             # (Nt, Ny, Nx)
    cine = np.transpose(cine, (1, 2, 0))        # (Ny, Nx, Nt)

    ds0 = infos[0][2]
    edges = cine_edges_from_dicom(ds0)
    if log_fn is not None:
        iop = np.array(ds0.ImageOrientationPatient, float)
        ipp = np.array(ds0.ImagePositionPatient, float)
        ps = np.array(ds0.PixelSpacing, float)
        X = _unit(iop[:3])
        Y = _unit(iop[3:])
        Z = _unit(np.cross(X, Y))
        x_len = float(np.linalg.norm(edges[:3, 0]))
        y_len = float(np.linalg.norm(edges[:3, 1]))
        log_fn(f"[cine] IOP={iop}")
        log_fn(f"[cine] IPP={ipp}")
        log_fn(f"[cine] PixelSpacing={ps}")
        log_fn(f"[cine] edges col_len={x_len:.4f} row_len={y_len:.4f}")
        log_fn(
            f"[cine] unit_check |X|={np.linalg.norm(X):.3f} "
            f"|Y|={np.linalg.norm(Y):.3f} |Z|={np.linalg.norm(Z):.3f}"
        )
        log_fn(f"[cine] cross(X,Y)={Z}")
    meta = {
        "IPP": np.array(ds0.ImagePositionPatient, float),
        "IOP": np.array(ds0.ImageOrientationPatient, float),
        "PixelSpacing": np.array(ds0.PixelSpacing, float),
        "edges": edges,
        "axis_map": infer_axis_map_from_iop_ipp(np.array(ds0.ImageOrientationPatient, float)),
    }
    return cine, meta


# ============================
# physics
# ============================

def compute_ke(vel, rho=1060.0):
    return 0.5 * rho * np.sum(vel ** 2, axis=3)


def compute_vorticity(vel, dx, dy, dz):
    dx, dy, dz = dx * 1e-3, dy * 1e-3, dz * 1e-3
    Ny, Nx, Nz, _, Nt = vel.shape
    vort = np.zeros((Ny, Nx, Nz, 3, Nt), np.float32)
    vortmag = np.zeros((Ny, Nx, Nz, Nt), np.float32)

    for t in range(Nt):
        vx = median_filter(vel[..., 0, t], 3)
        vy = median_filter(vel[..., 1, t], 3)
        vz = median_filter(vel[..., 2, t], 3)

        dvx_dy, dvx_dx, dvx_dz = np.gradient(vx, dy, dx, dz)
        dvy_dy, dvy_dx, dvy_dz = np.gradient(vy, dy, dx, dz)
        dvz_dy, dvz_dx, dvz_dz = np.gradient(vz, dy, dx, dz)

        vort[..., 0, t] = dvz_dy - dvy_dz
        vort[..., 1, t] = dvx_dz - dvz_dx
        vort[..., 2, t] = dvy_dx - dvx_dy
        vortmag[..., t] = np.linalg.norm(vort[..., :, t], axis=3)

    return vort, vortmag


# ============================
# UI
# ============================

class PackBuilder(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MV Pack Builder")

        self.mr = ""
        self.dcm4d = ""
        self.cines = []

        self.btn_mr = QtWidgets.QPushButton("Select mrStruct folder")
        self.lbl_mr = QtWidgets.QLabel("mrStruct: -")

        self.btn_4d = QtWidgets.QPushButton("Select 4D DICOM folder")
        self.lbl_4d = QtWidgets.QLabel("4D DICOM: -")

        self.btn_add_cine = QtWidgets.QPushButton("Add cine folder")
        self.lst_cine = QtWidgets.QListWidget()

        self.btn_build = QtWidgets.QPushButton("Build mvpack.h5")

        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)

        lay = QtWidgets.QVBoxLayout(self)
        for w in (
            self.btn_mr,
            self.lbl_mr,
            self.btn_4d,
            self.lbl_4d,
            self.btn_add_cine,
            self.lst_cine,
            self.btn_build,
            self.logbox,
        ):
            lay.addWidget(w)

        self.btn_mr.clicked.connect(self.sel_mr)
        self.btn_4d.clicked.connect(self.sel_4d)
        self.btn_add_cine.clicked.connect(self.add_cine)
        self.btn_build.clicked.connect(self.build)

    def log(self, s):
        self.logbox.appendPlainText(s)

    def sel_mr(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self)
        if d:
            self.mr = d
            self.lbl_mr.setText(f"mrStruct: {d}")

    def sel_4d(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self)
        if d:
            self.dcm4d = d
            self.lbl_4d.setText(f"4D DICOM: {d}")

    def add_cine(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not d:
            return
        tag, ok = QtWidgets.QInputDialog.getText(self, "cine tag", "2CH / 3CH / 4CH")
        if ok:
            self.cines.append((tag, d))
            self.lst_cine.addItem(f"{tag}: {d}")

    def build(self):
        self.log("=== BUILD START ===")
        self.log(f"mrStruct: {self.mr}")
        self.log(f"4D DICOM: {self.dcm4d}")
        self.log(f"cine count: {len(self.cines)}")

        mag, mag_meta = load_mrstruct(os.path.join(self.mr, "mag_struct.mat"))
        vel, vel_meta = load_mrstruct(os.path.join(self.mr, "vel_struct.mat"))

        geom = estimate_volume_geom(self.dcm4d)

        # prefer mrStruct edges if present
        edges = None
        vox = None
        for meta in (vel_meta, mag_meta):
            if meta is None:
                continue
            if edges is None and meta.get("edges") is not None:
                edges = meta["edges"]
            if vox is None and meta.get("vox") is not None:
                vox = meta["vox"]

        if edges is not None:
            edges = np.asarray(edges, float)
            if edges.shape == (3, 4):
                edges = np.vstack([edges, np.array([0.0, 0.0, 0.0, 1.0])])
            if edges.shape == (4, 4):
                # mrStruct.edges는 최상위 기준이다 (edges만 신뢰)
                geom["edges"] = edges
                geom["A"] = edges[0:3, 0:3]
                geom["orgn4"] = edges[0:3, 3]
                self.log("[info] using mrStruct.edges for volume geometry.")
            else:
                self.log(f"[warn] Unexpected mrStruct edges shape: {edges.shape}, using DICOM geometry.")
        else:
            self.log("[warn] mrStruct edges missing; using DICOM-derived edges.")

        # voxel spacing override if provided
        if vox is not None:
            vox = np.asarray(vox, float).reshape(-1)
            if vox.size >= 2:
                geom["PixelSpacing"] = vox[:2]
            if vox.size >= 3:
                geom["sliceStep"] = np.array([vox[2]])

        ke = compute_ke(vel)
        vort, vortmag = compute_vorticity(
            vel,
            geom["PixelSpacing"][1],
            geom["PixelSpacing"][0],
            geom["sliceStep"][0],
        )

        out = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not out:
            return

        out_path = os.path.join(out, "mvpack.h5")
        with h5py.File(out_path, "w") as f:
            g = f.create_group("data")
            g["vel"] = vel
            g["mag"] = mag
            g["ke"] = ke
            g["vort"] = vort
            g["vortmag"] = vortmag

            gg = f.create_group("geom")
            gg["edges"] = geom["edges"]
            gg["orgn4"] = geom["orgn4"]
            gg["A"] = geom["A"]
            gg["PixelSpacing"] = geom["PixelSpacing"]
            gg["sliceStep"] = geom["sliceStep"]
            gg["IOP"] = geom["IOP"]
            gg["IPPs"] = geom["IPPs"]
            gg["slice_positions"] = geom["slice_positions"]
            gg["slice_order"] = geom["slice_order"]
            gg.attrs["axis_map_json"] = json.dumps(_to_jsonable(geom["axis_map"]))

            gc = f.create_group("cine")
            for tag, folder in self.cines:
                cine, meta = read_cine(folder, log_fn=self.log)
                gt = gc.create_group(tag.lower())
                gt["cineI"] = cine
                gt["IPP"] = meta["IPP"]
                gt["IOP"] = meta["IOP"]
                gt["PixelSpacing"] = meta["PixelSpacing"]
                gt["edges"] = meta["edges"]
                gt.attrs["axis_map_json"] = json.dumps(_to_jsonable(meta["axis_map"]))
                self.log(f"cine saved: {tag}")

        self.log(f"DONE -> {out_path}")
        QtWidgets.QMessageBox.information(self, "Build Complete", "mvpack.h5 saved successfully.")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = PackBuilder()
    w.resize(900, 650)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
