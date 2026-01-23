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
    row = _unit(iop6[:3])
    col = _unit(iop6[3:])
    slc = _unit(np.cross(row, col))

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
    iop = np.array(ds0.ImageOrientationPatient, float)
    ipp = np.array(ds0.ImagePositionPatient, float)
    ps = np.array(ds0.PixelSpacing, float)
    slice_step = float(getattr(ds0, "SliceThickness", 1.0) or 1.0)

    X = iop[:3]
    Y = iop[3:]

    edges = np.eye(4, dtype=np.float64)
    edges[:3, 0] = X * ps[1]
    edges[:3, 1] = Y * ps[0]
    edges[:3, 2] = np.cross(X, Y) * slice_step
    edges[:3, 3] = ipp

    t = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    edges[:3, :3] = edges[:3, :3] @ t
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
    ds0 = infos[0][2]

    # mvpack_builder_v1_2.py  estimate_volume_geom()
    
    orgn4 = np.array(ds0.ImagePositionPatient, float)
    iop   = np.array(ds0.ImageOrientationPatient, float)
    ps    = np.array(ds0.PixelSpacing, float)
    
    # DICOM convention
    col = _unit(iop[0:3])   # +j direction
    row = _unit(iop[3:6])   # +i direction
    
    # match geometry.py per-slice branch: slc_dir = cross(row_dir, col_dir)
    slc = np.cross(row, col)
    nslc = np.linalg.norm(slc)
    slc = slc / nslc if nslc > 1e-6 else np.array([0.0, 0.0, 1.0])
    
    ipps = np.array([np.array(ds.ImagePositionPatient, float) for _, _, ds in infos])
    proj = ipps @ slc
    diffs = np.diff(np.sort(proj))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[np.abs(diffs) > 1e-6]
    dz = float(np.median(diffs)) if diffs.size else float(ps[0])
    dz = max(dz, 1e-3)
    
    A = np.column_stack([
        col * ps[1],   # col step
        row * ps[0],   # row step
        slc * dz,      # slice step
    ])


    if np.linalg.matrix_rank(A) < 3:
        raise RuntimeError(
            "Invalid volume geometry: A is singular "
            "(check IOP / IPP / slice spacing)"
        )

    return {
        "orgn4": orgn4,
        "A": A,
        "PixelSpacing": ps,
        "sliceStep": np.array([dz]),
        "IOP": iop,
        "IPPs": ipps,
        "slice_positions": proj,
        "slice_order": np.argsort(proj),
        "axis_map": infer_axis_map_from_iop_ipp(iop, ipps),
    }


def read_cine(folder):
    infos = read_dicom_sorted(folder)
    frames = [pydicom.dcmread(p, force=True).pixel_array.T for _, p, _ in infos]
    cine = np.stack(frames, axis=0)
    cine = np.transpose(cine, (1, 2, 0))  # (Ny, Nx, Nt)

    ds0 = infos[0][2]
    meta = {
        "IPP": np.array(ds0.ImagePositionPatient, float),
        "IOP": np.array(ds0.ImageOrientationPatient, float),
        "PixelSpacing": np.array(ds0.PixelSpacing, float),
        "edges": cine_edges_from_dicom(ds0),
        "axis_map": infer_axis_map_from_iop_ipp(
            np.array(ds0.ImageOrientationPatient, float)
        ),
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
        edges = None
        edges_from_mrstruct = False
        vox = None
        for meta in (vel_meta, mag_meta):
            if meta is None:
                continue
            if edges is None and meta.get("edges") is not None:
                edges = meta["edges"]
                edges_from_mrstruct = True
            if vox is None and meta.get("vox") is not None:
                vox = meta["vox"]

        if edges is not None:
            edges = np.asarray(edges, float)
            if edges.shape in {(4, 4), (3, 4)}:
                if edges_from_mrstruct:
                    # MATLAB mrStruct.edges store row/col due to transposed image
                    # storage; swap to align with reslice_plane_fixedN() [col,row,slice].
                    edges[:3, [0, 1]] = edges[:3, [1, 0]]
                geom["A"] = edges[0:3, 0:3]
                geom["orgn4"] = edges[0:3, 3]
            else:
                self.log(f"[warn] Unexpected edges shape: {edges.shape}, using DICOM geometry.")
                edges = None
        else:
            self.log("[warn] mrStruct edges missing; using DICOM geometry.")

        if vox is not None:
            vox = np.asarray(vox, float).reshape(-1)
            if vox.size >= 2:
                geom["PixelSpacing"] = vox[:2]
            if vox.size >= 3:
                geom["sliceStep"] = np.array([vox[2]])

        if edges is not None:
            geom["edges"] = edges
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

        with h5py.File(os.path.join(out, "mvpack.h5"), "w") as f:
            g = f.create_group("data")
            g["vel"] = vel
            g["mag"] = mag
            g["ke"] = ke
            g["vort"] = vort
            g["vortmag"] = vortmag

            gg = f.create_group("geom")
            gg["orgn4"] = geom["orgn4"]
            gg["A"] = geom["A"]
            if "edges" in geom:
                gg["edges"] = geom["edges"]
            gg["PixelSpacing"] = geom["PixelSpacing"]
            gg["sliceStep"] = geom["sliceStep"]
            gg["IOP"] = geom["IOP"]
            gg["IPPs"] = geom["IPPs"]
            gg["slice_positions"] = geom["slice_positions"]
            gg["slice_order"] = geom["slice_order"]
            gg.attrs["axis_map_json"] = json.dumps(_to_jsonable(geom["axis_map"]))

            gc = f.create_group("cine")
            for tag, folder in self.cines:
                cine, meta = read_cine(folder)
                gt = gc.create_group(tag)
                gt["cineI"] = cine
                gt["IPP"] = meta["IPP"]
                gt["IOP"] = meta["IOP"]
                gt["PixelSpacing"] = meta["PixelSpacing"]
                gt["edges"] = meta["edges"]
                gt.attrs["axis_map_json"] = json.dumps(_to_jsonable(meta["axis_map"]))
                self.log(f"cine saved: {tag}")

        self.log("DONE")
        QtWidgets.QMessageBox.information(self, "Build Complete", "mvpack.h5 saved successfully.")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = PackBuilder()
    w.resize(900, 650)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
