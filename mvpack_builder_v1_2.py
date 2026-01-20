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
    col = _unit(iop6[:3])
    row = _unit(iop6[3:])
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


def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return x


def axis_dir_label(vec, lps_to_ras=False):
    axes = ["X", "Y", "Z"]
    vec = np.asarray(vec, float).reshape(3)
    if lps_to_ras:
        vec = vec.copy()
        vec[0] *= -1
        vec[1] *= -1
    idx = int(np.argmax(np.abs(vec)))
    sign = "+" if vec[idx] >= 0 else "-"
    return f"{axes[idx]}{sign}"


def axis_map_summary(axis_map, lps_to_ras=False):
    return {
        "Rows": axis_dir_label(axis_map["Rows"], lps_to_ras=lps_to_ras),
        "Columns": axis_dir_label(axis_map["Columns"], lps_to_ras=lps_to_ras),
        "Slices": axis_dir_label(axis_map["Slices"], lps_to_ras=lps_to_ras),
    }


# ============================
# IO
# ============================

def load_mrstruct(path):
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    ms = md["mrStruct"]
    return np.array(ms.dataAy), None


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

    orgn4 = np.array(ds0.ImagePositionPatient, float)
    iop = np.array(ds0.ImageOrientationPatient, float)
    ps = np.array(ds0.PixelSpacing, float)

    col = _unit(iop[:3])
    row = _unit(iop[3:])

    slc = np.cross(row, col)
    nslc = np.linalg.norm(slc)

    if nslc < 1e-6:
        slc = np.array([0.0, 0.0, 1.0])
    else:
        slc = slc / nslc

    ipps = np.array([np.array(ds.ImagePositionPatient, float) for _, _, ds in infos])
    proj = ipps @ slc
    diffs = np.diff(np.sort(proj))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[np.abs(diffs) > 1e-6]

    if diffs.size == 0:
        dz = float(ps[0])
    else:
        dz = float(np.median(diffs))

    dz = max(dz, 1e-3)

    A = np.column_stack([
        col * ps[1],
        row * ps[0],
        slc * dz,
    ])

    if np.linalg.matrix_rank(A) < 3:
        raise RuntimeError(
            "Invalid volume geometry: A is singular "
            "(check IOP / IPP / slice spacing)"
        )

    order = np.argsort(proj)
    slice_positions = proj[order]

    return {
        "orgn4": orgn4,
        "A": A,
        "PixelSpacing": ps,
        "sliceStep": np.array([dz]),
        "axis_map": infer_axis_map_from_iop_ipp(iop, ipps),
        "IOP": iop,
        "IPP0": orgn4,
        "IPPs": ipps,
        "slice_positions": slice_positions,
        "slice_order": order,
    }


def read_cine(folder):
    infos = read_dicom_sorted(folder)
    frames = [pydicom.dcmread(p, force=True).pixel_array for _, p, _ in infos]
    cine = np.stack(frames, axis=0)  # (Nt, Ny, Nx)

    ds0 = infos[0][2]
    meta = {
        "IPP": np.array(ds0.ImagePositionPatient, float),
        "IOP": np.array(ds0.ImageOrientationPatient, float),
        "PixelSpacing": np.array(ds0.PixelSpacing, float),
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
        self._last_dir = os.getcwd()

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
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select mrStruct folder", self._last_dir)
        if d:
            self.mr = d
            self._last_dir = d
            self.lbl_mr.setText(f"mrStruct: {d}")

    def sel_4d(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select 4D DICOM folder", self._last_dir)
        if d:
            self.dcm4d = d
            self._last_dir = d
            self.lbl_4d.setText(f"4D DICOM: {d}")

    def add_cine(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Add cine folder", self._last_dir)
        if not d:
            return
        tag, ok = QtWidgets.QInputDialog.getText(self, "cine tag", "2CH / 3CH / 4CH")
        if ok:
            self.cines.append((tag, d))
            self._last_dir = d
            self.lst_cine.addItem(f"{tag}: {d}")

    def build(self):
        self.log("=== BUILD START ===")
        self.log(f"mrStruct: {self.mr}")
        self.log(f"4D DICOM: {self.dcm4d}")
        self.log(f"cine count: {len(self.cines)}")

        mag, _ = load_mrstruct(os.path.join(self.mr, "mag_struct.mat"))
        vel, _ = load_mrstruct(os.path.join(self.mr, "vel_struct.mat"))

        geom = estimate_volume_geom(self.dcm4d)
        ke = compute_ke(vel)
        vort, vortmag = compute_vorticity(
            vel,
            geom["PixelSpacing"][1],
            geom["PixelSpacing"][0],
            geom["sliceStep"][0],
        )
        geom_axes = axis_map_summary(geom["axis_map"], lps_to_ras=True)
        self.log(
            "4D DICOM axes: "
            f"Rows={geom_axes['Rows']} Columns={geom_axes['Columns']} Slices={geom_axes['Slices']}"
        )

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save mvpack",
            os.path.join(self._last_dir, "mvpack.h5"),
            "HDF5 Files (*.h5)",
        )
        if not out_path:
            return
        out_dir = os.path.dirname(out_path)
        if out_dir:
            self._last_dir = out_dir

        with h5py.File(out_path, "w") as f:
            g = f.create_group("data")
            g["vel"] = vel
            g["mag"] = mag
            g["ke"] = ke
            g["vort"] = vort
            g["vortmag"] = vortmag

            gg = f.create_group("geom")
            gg["orgn4"] = geom["orgn4"]
            gg["A"] = geom["A"]
            gg["PixelSpacing"] = geom["PixelSpacing"]
            gg["sliceStep"] = geom["sliceStep"]
            gg.attrs["axis_map_json"] = json.dumps(_to_jsonable(geom["axis_map"]))
            gg["IOP"] = geom["IOP"]
            gg["IPP0"] = geom["IPP0"]
            gg["IPPs"] = geom["IPPs"]
            gg["slice_positions"] = geom["slice_positions"]
            gg["slice_order"] = geom["slice_order"]

            gc = f.create_group("cine")
            for tag, folder in self.cines:
                cine, meta = read_cine(folder)
                cine_axes = axis_map_summary(meta["axis_map"], lps_to_ras=True)
                self.log(
                    "cine meta "
                    f"{tag}: IOP={np.array2string(meta['IOP'], precision=6, separator=',')}, "
                    f"IPP={np.array2string(meta['IPP'], precision=6, separator=',')}, "
                    f"Rows={cine_axes['Rows']} Columns={cine_axes['Columns']} Slices={cine_axes['Slices']}"
                )
                gt = gc.create_group(tag)
                gt["cineI"] = cine
                gt["IPP"] = meta["IPP"]
                gt["IOP"] = meta["IOP"]
                gt["PixelSpacing"] = meta["PixelSpacing"]
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
