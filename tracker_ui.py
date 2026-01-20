import json
import os
from typing import Dict, Tuple, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from geometry import reslice_plane_fixedN, cine_line_to_patient_xyz
from stl_conversion import convert_plane_to_stl
from mvpack_io import MVPack, CineGeom
from plot_widgets import PlotCanvas, _WheelToSliderFilter
from roi_utils import closed_spline_xy, polygon_mask
from tracking_state import (
    mvtrack_path_for_folder,
    find_mvtrack_files,
    save_tracking_state_h5,
    load_tracking_state_h5,
)


class ValveTracker(QtWidgets.QMainWindow):
    def __init__(
        self,
        pack: MVPack,
        work_folder: str,
        tracking_path: Optional[str] = None,
        restore_state: bool = True,
    ):
        super().__init__()
        self.setWindowTitle("MV tracking 4D (mvpack + MVtrack state)")

        self.pack = pack
        self.work_folder = work_folder
        self.tracking_path = tracking_path

        self.Nt = int(pack.pcmra.shape[3])
        self.Npix = 192

        keys = list(pack.cine_planes.keys())
        if "2ch" in pack.cine_planes:
            self.active_cine_key = "2ch"
        else:
            self.active_cine_key = sorted(keys)[0]

        self.line_norm = [None] * self.Nt
        self.roi_state = [None] * self.Nt
        self.roi_locked = [False] * self.Nt

        self.metrics_Q = np.full(self.Nt, np.nan, dtype=np.float64)
        self.metrics_Vpk = np.full(self.Nt, np.nan, dtype=np.float64)
        self.metrics_Vmn = np.full(self.Nt, np.nan, dtype=np.float64)
        self.metrics_KE = np.full(self.Nt, np.nan, dtype=np.float64)
        self.metrics_VortPk = np.full(self.Nt, np.nan, dtype=np.float64)
        self.metrics_VortMn = np.full(self.Nt, np.nan, dtype=np.float64)

        self._voxel_volume_m3 = abs(float(np.linalg.det(self.pack.geom.A))) * 1e-9

        self.poly_roi_vel = None
        self.poly_roi_pcm = None
        self.spline_curve_pcm = None
        self.spline_curve_vel = None

        self.edit_mode = False
        self._syncing_poly = False
        self._cur_phase = None
        self._syncing_cine_line = False

        self._view_ranges = {"pcmra": None, "vel": None}
        self._restoring_view = False
        self._updating_image = False
        self._roi_clipboard = None
        self._line_clipboard = None
        self.lock_label_pcm = None
        self.lock_label_vel = None
        self._restored_state = False

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)

        layout = QtWidgets.QGridLayout(cw)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 0)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 3)

        # LEFT: cine selector + multi cine
        self.cine_views: Dict[str, pg.ImageView] = {}
        self.cine_line_rois: Dict[str, pg.LineSegmentROI] = {}

        left_box = QtWidgets.QVBoxLayout()
        layout.addLayout(left_box, 0, 0, 2, 1)

        self.cine_selector = QtWidgets.QComboBox()
        left_box.addWidget(self.cine_selector)

        cine_level_row = QtWidgets.QHBoxLayout()
        cine_level_row.addWidget(QtWidgets.QLabel("Cine Min"))
        self.cine_min = QtWidgets.QDoubleSpinBox()
        self.cine_min.setDecimals(4)
        self.cine_min.setRange(-1e9, 1e9)
        cine_level_row.addWidget(self.cine_min)
        cine_level_row.addWidget(QtWidgets.QLabel("Max"))
        self.cine_max = QtWidgets.QDoubleSpinBox()
        self.cine_max.setDecimals(4)
        self.cine_max.setRange(-1e9, 1e9)
        cine_level_row.addWidget(self.cine_max)
        self.btn_cine_auto_levels = QtWidgets.QPushButton("Auto")
        cine_level_row.addWidget(self.btn_cine_auto_levels)
        self.btn_cine_apply_levels = QtWidgets.QPushButton("Apply")
        cine_level_row.addWidget(self.btn_cine_apply_levels)
        cine_level_row.addStretch(1)
        left_box.addLayout(cine_level_row)

        def _cine_sort_key(name: str):
            n = name.lower()
            if "2ch" in n:
                return 0
            if "3ch" in n:
                return 1
            if "4ch" in n:
                return 2
            return 9

        cine_keys = sorted(self.pack.cine_planes.keys(), key=_cine_sort_key)
        self.cine_selector.addItems(cine_keys)
        self.cine_selector.setCurrentText(self.active_cine_key)
        self.cine_selector.currentTextChanged.connect(self.on_active_cine_changed)

        for k in cine_keys:
            view = pg.ImageView()
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.ui.histogram.setFixedWidth(90)
            view.ui.histogram.hide()
            left_box.addWidget(view, stretch=1)

            roi = pg.LineSegmentROI([[80, 80], [110, 60]], pen=pg.mkPen("y", width=3))
            view.getView().addItem(roi)

            roi.sigRegionChanged.connect(lambda _=None, kk=k: self.on_cine_line_changed_live(kk))
            roi.sigRegionChangeFinished.connect(lambda _=None, kk=k: self.on_cine_line_changed_finished(kk))
            view.getView().scene().sigMouseClicked.connect(lambda _evt=None, kk=k: self.on_active_cine_changed(kk))

            self.cine_views[k] = view
            self.cine_line_rois[k] = roi

        # RIGHT: pcmra / vel
        self.pcmra_view = pg.ImageView()
        self.pcmra_view.ui.roiBtn.hide()
        self.pcmra_view.ui.menuBtn.hide()
        self.pcmra_view.ui.histogram.setFixedWidth(90)
        self.pcmra_view.ui.histogram.hide()

        pcmra_box = QtWidgets.QVBoxLayout()
        layout.addLayout(pcmra_box, 0, 1)

        pcmra_level_row = QtWidgets.QHBoxLayout()
        pcmra_box.addLayout(pcmra_level_row)
        pcmra_level_row.addWidget(QtWidgets.QLabel("PCMRA Min"))
        self.pcmra_min = QtWidgets.QDoubleSpinBox()
        self.pcmra_min.setDecimals(4)
        self.pcmra_min.setRange(-1e9, 1e9)
        pcmra_level_row.addWidget(self.pcmra_min)
        pcmra_level_row.addWidget(QtWidgets.QLabel("Max"))
        self.pcmra_max = QtWidgets.QDoubleSpinBox()
        self.pcmra_max.setDecimals(4)
        self.pcmra_max.setRange(-1e9, 1e9)
        pcmra_level_row.addWidget(self.pcmra_max)
        self.btn_pcmra_auto = QtWidgets.QPushButton("Auto")
        pcmra_level_row.addWidget(self.btn_pcmra_auto)
        self.btn_pcmra_apply = QtWidgets.QPushButton("Apply")
        pcmra_level_row.addWidget(self.btn_pcmra_apply)
        pcmra_level_row.addStretch(1)

        pcmra_box.addWidget(self.pcmra_view, stretch=1)

        self.vel_view = pg.ImageView()
        self.vel_view.ui.roiBtn.hide()
        self.vel_view.ui.menuBtn.hide()
        self.vel_view.ui.histogram.setFixedWidth(90)
        self.vel_view.ui.histogram.hide()

        vel_box = QtWidgets.QVBoxLayout()
        layout.addLayout(vel_box, 0, 2)

        display_row = QtWidgets.QHBoxLayout()
        vel_box.addLayout(display_row)
        display_row.addWidget(QtWidgets.QLabel("Display"))
        self.display_selector = QtWidgets.QComboBox()
        self.display_selector.addItems(["Velocity", "Kinetic energy", "Vorticity"])
        display_row.addWidget(self.display_selector)
        display_row.addStretch(1)
        self._current_display_mode = self.display_selector.currentText()

        level_row = QtWidgets.QHBoxLayout()
        vel_box.addLayout(level_row)
        level_row.addWidget(QtWidgets.QLabel("Min"))
        self.level_min = QtWidgets.QDoubleSpinBox()
        self.level_min.setDecimals(4)
        self.level_min.setRange(-1e9, 1e9)
        level_row.addWidget(self.level_min)
        level_row.addWidget(QtWidgets.QLabel("Max"))
        self.level_max = QtWidgets.QDoubleSpinBox()
        self.level_max.setDecimals(4)
        self.level_max.setRange(-1e9, 1e9)
        level_row.addWidget(self.level_max)
        self.btn_auto_levels = QtWidgets.QPushButton("Auto")
        level_row.addWidget(self.btn_auto_levels)
        self.btn_apply_levels = QtWidgets.QPushButton("Apply")
        level_row.addWidget(self.btn_apply_levels)
        level_row.addStretch(1)

        vel_box.addWidget(self.vel_view, stretch=1)
        self.display_selector.currentTextChanged.connect(self.on_display_changed)

        self.lock_label_pcm = pg.TextItem("LOCK", color=(255, 60, 60))
        self.lock_label_vel = pg.TextItem("LOCK", color=(255, 60, 60))
        self.lock_label_pcm.setVisible(False)
        self.lock_label_vel.setVisible(False)
        self.pcmra_view.getView().addItem(self.lock_label_pcm)
        self.vel_view.getView().addItem(self.lock_label_vel)

        self._display_colormaps = {}
        for name, cmap in [("Velocity", "jet"), ("Kinetic energy", "viridis"), ("Vorticity", "plasma")]:
            try:
                self._display_colormaps[name] = pg.colormap.get(cmap, source="matplotlib")
            except Exception:
                self._display_colormaps[name] = None
        self._display_levels: Dict[str, Tuple[Optional[float], Optional[float]]] = {
            "Velocity": (None, None),
            "Kinetic energy": (None, None),
            "Vorticity": (None, None),
        }
        self._vel_auto_once: Dict[str, bool] = {
            "Velocity": True,
            "Kinetic energy": True,
            "Vorticity": True,
        }
        self._pcmra_levels: Tuple[Optional[float], Optional[float]] = (None, None)
        self._cine_levels: Tuple[Optional[float], Optional[float]] = (None, None)
        self._pcmra_auto_once = True
        self._cine_auto_once = True

        # bottom right
        bottom_right = QtWidgets.QVBoxLayout()
        layout.addLayout(bottom_right, 1, 1, 1, 2)

        chart_log_row = QtWidgets.QHBoxLayout()
        bottom_right.addLayout(chart_log_row, stretch=1)

        chart_box = QtWidgets.QVBoxLayout()
        chart_log_row.addLayout(chart_box, stretch=3)

        chart_row = QtWidgets.QHBoxLayout()
        chart_row.addWidget(QtWidgets.QLabel("Chart"))
        self.chart_selector = QtWidgets.QComboBox()
        self.chart_selector.addItems(
            [
                "Flow rate (mL/s)",
                "Peak velocity (m/s)",
                "Mean velocity (m/s)",
                "Kinetic energy (uJ)",
                "Peak vorticity (1/s)",
                "Mean vorticity (1/s)",
            ]
        )
        chart_row.addWidget(self.chart_selector)
        chart_row.addStretch(1)
        chart_box.addLayout(chart_row)

        self.plot = PlotCanvas()
        chart_box.addWidget(self.plot, stretch=1)
        self.chart_selector.currentTextChanged.connect(self.update_plot_for_selection)
        self.plot.set_phase_callback(self.on_plot_phase_selected)

        log_box = QtWidgets.QVBoxLayout()
        chart_log_row.addLayout(log_box, stretch=2)
        log_box.addWidget(QtWidgets.QLabel("Log"))
        self.memo = QtWidgets.QPlainTextEdit()
        self.memo.setReadOnly(True)
        log_box.addWidget(self.memo, stretch=1)

        stats_row = QtWidgets.QHBoxLayout()
        bottom_right.addLayout(stats_row)

        self.lbl_phase = QtWidgets.QLabel("Phase: 1")
        self.lbl_Q = QtWidgets.QLabel("Flow rate (mL/s): -")
        self.lbl_Vpk = QtWidgets.QLabel("Peak velocity (m/s): -")
        self.lbl_Vmn = QtWidgets.QLabel("Mean velocity (m/s): -")
        self.lbl_KE = QtWidgets.QLabel("Kinetic energy (uJ): -")
        self.lbl_VortPk = QtWidgets.QLabel("Peak vorticity (1/s): -")
        self.lbl_VortMn = QtWidgets.QLabel("Mean vorticity (1/s): -")

        for w in [self.lbl_phase, self.lbl_Q, self.lbl_Vpk, self.lbl_Vmn, self.lbl_KE, self.lbl_VortPk, self.lbl_VortMn]:
            stats_row.addWidget(w)
        stats_row.addStretch(1)

        btn_row = QtWidgets.QHBoxLayout()
        bottom_right.addLayout(btn_row)

        self.btn_compute = QtWidgets.QPushButton("Compute current")
        self.btn_all = QtWidgets.QPushButton("Compute all")
        self.btn_edit = QtWidgets.QPushButton("Edit ROI: OFF")
        self.btn_copy = QtWidgets.QPushButton("Copy all phases")
        self.btn_save = QtWidgets.QPushButton("Save to MVtrack.h5")
        self.btn_convert_stl = QtWidgets.QPushButton("Convert STL")
        self.btn_roi_copy = QtWidgets.QPushButton("Copy ROI")
        self.btn_roi_paste = QtWidgets.QPushButton("Paste ROI")
        self.btn_roi_forward = QtWidgets.QPushButton("Copy ROI forward")
        self.btn_roi_lock = QtWidgets.QPushButton("Lock ROI: OFF")

        btn_row.addWidget(self.btn_compute)
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_roi_copy)
        btn_row.addWidget(self.btn_roi_paste)
        btn_row.addWidget(self.btn_roi_forward)
        btn_row.addWidget(self.btn_roi_lock)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_convert_stl)
        btn_row.addStretch(1)

        self.btn_compute.clicked.connect(self.compute_current)
        self.btn_all.clicked.connect(self.compute_all)
        self.btn_edit.clicked.connect(self.toggle_edit)
        self.btn_copy.clicked.connect(self.copy_current_to_clipboard)
        self.btn_roi_copy.clicked.connect(self.copy_roi_state)
        self.btn_roi_paste.clicked.connect(self.paste_roi_state)
        self.btn_roi_forward.clicked.connect(self.copy_roi_forward)
        self.btn_roi_lock.clicked.connect(self.toggle_roi_lock)
        self.btn_save.clicked.connect(self.save_to_mvtrack_h5)
        self.btn_convert_stl.clicked.connect(self.convert_to_stl)
        self.btn_apply_levels.clicked.connect(self.apply_level_range)
        self.btn_auto_levels.clicked.connect(self.enable_auto_levels)
        self.btn_pcmra_apply.clicked.connect(self.apply_pcmra_levels)
        self.btn_pcmra_auto.clicked.connect(self.enable_pcmra_auto)
        self.btn_cine_apply_levels.clicked.connect(self.apply_cine_levels)
        self.btn_cine_auto_levels.clicked.connect(self.enable_cine_auto)
        self.level_min.valueChanged.connect(self.on_level_spin_changed)
        self.level_max.valueChanged.connect(self.on_level_spin_changed)

        copy_action = QtGui.QAction(self)
        copy_action.setShortcut(QtGui.QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_roi_state)
        self.addAction(copy_action)

        paste_action = QtGui.QAction(self)
        paste_action.setShortcut(QtGui.QKeySequence.Paste)
        paste_action.triggered.connect(self.paste_roi_state)
        self.addAction(paste_action)

        # slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(max(1, self.Nt))
        self.slider.setValue(1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTracking(True)
        self.slider.valueChanged.connect(self.on_phase_changed)
        layout.addWidget(self.slider, 2, 0, 1, 3)

        # wheel filter
        self._wheel_filter = _WheelToSliderFilter(self.slider)
        for v in self.cine_views.values():
            v.installEventFilter(self._wheel_filter)
        self.pcmra_view.installEventFilter(self._wheel_filter)
        self.vel_view.installEventFilter(self._wheel_filter)
        self.slider.installEventFilter(self._wheel_filter)

        # range save
        for k, view in self.cine_views.items():
            view.getView().sigRangeChanged.connect(lambda *_unused, kk=k: self._store_view_range(f"cine:{kk}"))
        self.pcmra_view.getView().sigRangeChanged.connect(lambda *_: self._store_view_range("pcmra"))
        self.vel_view.getView().sigRangeChanged.connect(lambda *_: self._store_view_range("vel"))
        self.pcmra_view.getView().sigRangeChanged.connect(lambda *_: self._update_lock_label_positions())
        self.vel_view.getView().sigRangeChanged.connect(lambda *_: self._update_lock_label_positions())

        line_ctrl_row = QtWidgets.QHBoxLayout()
        self.btn_line_copy = QtWidgets.QPushButton("Copy line")
        self.btn_line_paste = QtWidgets.QPushButton("Paste line")
        self.btn_line_forward = QtWidgets.QPushButton("Copy line forward")
        self.btn_plane_overlay = QtWidgets.QPushButton("Plane Overlay")
        line_ctrl_row.addWidget(self.btn_line_copy)
        line_ctrl_row.addWidget(self.btn_line_paste)
        line_ctrl_row.addWidget(self.btn_line_forward)
        line_ctrl_row.addWidget(self.btn_plane_overlay)
        line_ctrl_row.addStretch(1)
        left_box.addLayout(line_ctrl_row)
        self.btn_line_copy.clicked.connect(self.copy_line_state)
        self.btn_line_paste.clicked.connect(self.paste_line_state)
        self.btn_line_forward.clicked.connect(self.copy_line_forward)
        self.btn_plane_overlay.clicked.connect(self.show_plane_overlay)

        cine_xform_row = QtWidgets.QHBoxLayout()
        cine_xform_row.addWidget(QtWidgets.QLabel("Flip"))
        self.chk_cine_flip_x = QtWidgets.QCheckBox("X")
        self.chk_cine_flip_y = QtWidgets.QCheckBox("Y")
        self.chk_cine_flip_z = QtWidgets.QCheckBox("Z")
        cine_xform_row.addWidget(self.chk_cine_flip_x)
        cine_xform_row.addWidget(self.chk_cine_flip_y)
        cine_xform_row.addWidget(self.chk_cine_flip_z)
        cine_xform_row.addWidget(QtWidgets.QLabel("Swap"))
        self.cine_swap_selector = QtWidgets.QComboBox()
        self.cine_swap_selector.addItems(
            ["X Y Z", "X Z Y", "Y X Z", "Y Z X", "Z X Y", "Z Y X"]
        )
        cine_xform_row.addWidget(self.cine_swap_selector)
        self.btn_cine_apply = QtWidgets.QPushButton("Apply")
        cine_xform_row.addWidget(self.btn_cine_apply)
        cine_xform_row.addStretch(1)
        left_box.addLayout(cine_xform_row)
        self.btn_cine_apply.clicked.connect(self.on_cine_transform_changed)

        # try restore MVtrack.h5
        if restore_state:
            self.try_restore_state()

        self._update_cine_roi_visibility()
        self.set_phase(0)

    # ============================
    # Restore state
    # ============================
    def try_restore_state(self):
        st_path = self.tracking_path or mvtrack_path_for_folder(self.work_folder)
        if not os.path.exists(st_path):
            hits = find_mvtrack_files(self.work_folder)
            if hits:
                st_path = hits[-1]
        st = load_tracking_state_h5(st_path, self.Nt)
        if st is None:
            self.memo.appendPlainText("No existing MVtrack.h5 found. Starting fresh.")
            return
        self._restored_state = True

        self.line_norm = st.get("line_norm", self.line_norm)
        self.roi_state = st.get("roi_state", self.roi_state)
        self.roi_locked = st.get("roi_locked", self.roi_locked)

        Q = st.get("metrics_Q", None)
        Vpk = st.get("metrics_Vpk", None)
        Vmn = st.get("metrics_Vmn", None)
        KE = st.get("metrics_KE", None)
        VortPk = st.get("metrics_VortPk", None)
        VortMn = st.get("metrics_VortMn", None)
        if Q is not None:
            self.metrics_Q[: min(len(Q), self.Nt)] = Q[: min(len(Q), self.Nt)]
        if Vpk is not None:
            self.metrics_Vpk[: min(len(Vpk), self.Nt)] = Vpk[: min(len(Vpk), self.Nt)]
        if Vmn is not None:
            self.metrics_Vmn[: min(len(Vmn), self.Nt)] = Vmn[: min(len(Vmn), self.Nt)]
        if KE is not None:
            self.metrics_KE[: min(len(KE), self.Nt)] = KE[: min(len(KE), self.Nt)]
        if VortPk is not None:
            self.metrics_VortPk[: min(len(VortPk), self.Nt)] = VortPk[: min(len(VortPk), self.Nt)]
        if VortMn is not None:
            self.metrics_VortMn[: min(len(VortMn), self.Nt)] = VortMn[: min(len(VortMn), self.Nt)]

        active = st.get("active_cine_key", "")
        if active and active in self.pack.cine_planes:
            self.active_cine_key = active
            if hasattr(self, "cine_selector"):
                self.cine_selector.blockSignals(True)
                try:
                    self.cine_selector.setCurrentText(active)
                finally:
                    self.cine_selector.blockSignals(False)

        try:
            display_levels = json.loads(st.get("display_levels_json", "{}"))
            vel_auto_once = json.loads(st.get("vel_auto_once_json", "{}"))
            cine_levels = json.loads(st.get("cine_levels_json", "[null, null]"))
            pcmra_levels = json.loads(st.get("pcmra_levels_json", "[null, null]"))
            cine_flip = json.loads(st.get("cine_flip_json", "[false, false, false]"))
        except Exception:
            display_levels = {}
            vel_auto_once = {}
            cine_levels = [None, None]
            pcmra_levels = [None, None]
            cine_flip = [False, False, False]
        cine_auto_once = bool(st.get("cine_auto_once", 1))
        pcmra_auto_once = bool(st.get("pcmra_auto_once", 1))
        cine_swap = st.get("cine_swap", "X Y Z")

        for key, val in display_levels.items():
            if key in self._display_levels and isinstance(val, (list, tuple)) and len(val) == 2:
                self._display_levels[key] = (val[0], val[1])
        for key, val in vel_auto_once.items():
            if key in self._vel_auto_once:
                self._vel_auto_once[key] = bool(val)

        if isinstance(cine_levels, (list, tuple)) and len(cine_levels) == 2:
            self._cine_levels = (cine_levels[0], cine_levels[1])
        if isinstance(pcmra_levels, (list, tuple)) and len(pcmra_levels) == 2:
            self._pcmra_levels = (pcmra_levels[0], pcmra_levels[1])
        self._cine_auto_once = bool(cine_auto_once)
        self._pcmra_auto_once = bool(pcmra_auto_once)

        if isinstance(cine_flip, (list, tuple)) and len(cine_flip) == 3:
            self.chk_cine_flip_x.setChecked(bool(cine_flip[0]))
            self.chk_cine_flip_y.setChecked(bool(cine_flip[1]))
            self.chk_cine_flip_z.setChecked(bool(cine_flip[2]))
        if isinstance(cine_swap, str) and cine_swap in [self.cine_swap_selector.itemText(i) for i in range(self.cine_swap_selector.count())]:
            self.cine_swap_selector.setCurrentText(cine_swap)

        self._sync_level_controls_from_state()

        self.memo.appendPlainText(f"Restored tracking state from: {st_path}")

        self.update_plot_for_selection()

    # ============================
    # Cine helpers
    # ============================
    def _get_cine_frame(self, cine_key: str, t: int) -> np.ndarray:
        return self._get_cine_frame_raw(cine_key, t)

    def _get_cine_frame_raw(self, cine_key: str, t: int) -> np.ndarray:
        cine = self.pack.cine_planes[cine_key]["img"]
        if cine.ndim == 2:
            return cine.astype(np.float32)
        NtC = cine.shape[2]
        if NtC <= 1:
            return cine[:, :, 0].astype(np.float32)
        idx = int(round((t / max(self.Nt - 1, 1)) * (NtC - 1)))
        idx = int(np.clip(idx, 0, NtC - 1))
        return cine[:, :, idx].astype(np.float32)

    def _volume_axis_permutation(self) -> Tuple[int, int, int]:
        mapping = {
            "X Y Z": (0, 1, 2),
            "X Z Y": (0, 2, 1),
            "Y X Z": (1, 0, 2),
            "Y Z X": (1, 2, 0),
            "Z X Y": (2, 0, 1),
            "Z Y X": (2, 1, 0),
        }
        return mapping.get(self.cine_swap_selector.currentText(), (0, 1, 2))

    def _volume_axis_flips(self) -> np.ndarray:
        return np.array(
            [
                -1.0 if self.chk_cine_flip_x.isChecked() else 1.0,
                -1.0 if self.chk_cine_flip_y.isChecked() else 1.0,
                -1.0 if self.chk_cine_flip_z.isChecked() else 1.0,
            ],
            dtype=np.float64,
        )

    def _get_cine_geom_raw(self, cine_key: str) -> CineGeom:
        return self.pack.cine_planes[cine_key]["geom"]

    def _apply_volume_transform(self):
        perm = self._volume_axis_permutation()
        flips = self._volume_axis_flips()

        def _permute_volume(arr: np.ndarray) -> np.ndarray:
            if arr is None:
                return arr
            if arr.ndim < 3:
                return arr
            axes = list(range(arr.ndim))
            axes[:3] = [perm[0], perm[1], perm[2]]
            out = np.transpose(arr, axes)
            for axis in range(3):
                if flips[axis] < 0:
                    out = np.flip(out, axis=axis)
            return out

        self.pack.pcmra = _permute_volume(self.pack.pcmra)
        self.pack.vel = _permute_volume(self.pack.vel)
        if self.pack.ke is not None:
            self.pack.ke = _permute_volume(self.pack.ke)
        if self.pack.vortmag is not None:
            self.pack.vortmag = _permute_volume(self.pack.vortmag)

        Ny, Nx, Nz = self.pack.pcmra.shape[:3]
        row_vec = self.pack.geom.A[:, 1].copy()
        col_vec = self.pack.geom.A[:, 0].copy()
        slc_vec = self.pack.geom.A[:, 2].copy()
        basis = [row_vec, col_vec, slc_vec]
        new_row = basis[perm[0]]
        new_col = basis[perm[1]]
        new_slc = basis[perm[2]]

        orgn4 = self.pack.geom.orgn4.copy()
        if flips[0] < 0:
            orgn4 = orgn4 + new_row * (Ny - 1)
            new_row = -new_row
        if flips[1] < 0:
            orgn4 = orgn4 + new_col * (Nx - 1)
            new_col = -new_col
        if flips[2] < 0:
            orgn4 = orgn4 + new_slc * (Nz - 1)
            new_slc = -new_slc

        self.pack.geom.orgn4 = orgn4
        self.pack.geom.A = np.column_stack([new_col, new_row, new_slc])

    def on_cine_transform_changed(self):
        self._apply_volume_transform()
        if self._cur_phase is not None:
            cur = self._cur_phase
            self._cur_phase = None
            self.set_phase(cur)

    def copy_line_state(self):
        t = int(self.slider.value()) - 1
        self._sync_current_phase_state()
        st = self.line_norm[t]
        if st is None:
            return
        self._line_clipboard = np.array(st, dtype=np.float64).copy()
        self.memo.appendPlainText("Copied cine line.")

    def paste_line_state(self):
        if self._line_clipboard is None:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            self.memo.appendPlainText("ROI is locked for this phase.")
            return
        self.line_norm[t] = np.array(self._line_clipboard, dtype=np.float64).copy()
        self._cur_phase = None
        self.set_phase(t)
        self.compute_current(update_only=True)
        self.memo.appendPlainText("Pasted cine line.")

    def copy_line_forward(self):
        t = int(self.slider.value()) - 1
        self._sync_current_phase_state()
        st = self.line_norm[t]
        if st is None:
            return
        for tt in range(t + 1, self.Nt):
            if not self._is_roi_locked(tt):
                self.line_norm[tt] = np.array(st, dtype=np.float64).copy()
        self.memo.appendPlainText("Copied cine line forward.")

    def _roi_points_abs(self, roi: pg.LineSegmentROI) -> np.ndarray:
        hs = roi.getHandles()
        p = roi.pos()
        p0 = hs[0].pos() + p
        p1 = hs[1].pos() + p
        return np.array([[p0.x(), p0.y()], [p1.x(), p1.y()]], dtype=np.float64)

    def _apply_line_abs_to_roi(self, roi: pg.LineSegmentROI, pts_xy: np.ndarray, H: int, W: int):
        pts = np.array(pts_xy, dtype=np.float64)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        pos = pts[0].copy()
        local = pts - pos[None, :]

        st = roi.getState()
        st["pos"] = (float(pos[0]), float(pos[1]))
        st["points"] = [(float(local[0, 0]), float(local[0, 1])), (float(local[1, 0]), float(local[1, 1]))]
        roi.setState(st)

    def _abs_to_norm_line(self, pts_xy: np.ndarray, H: int, W: int) -> np.ndarray:
        denom_x = max(W - 1, 1)
        denom_y = max(H - 1, 1)
        out = np.zeros((2, 2), dtype=np.float64)
        out[:, 0] = pts_xy[:, 0] / denom_x
        out[:, 1] = pts_xy[:, 1] / denom_y
        out = np.clip(out, 0.0, 1.0)
        return out

    def _norm_to_abs_line(self, pts_norm: np.ndarray, H: int, W: int) -> np.ndarray:
        denom_x = max(W - 1, 1)
        denom_y = max(H - 1, 1)
        out = np.zeros((2, 2), dtype=np.float64)
        out[:, 0] = pts_norm[:, 0] * denom_x
        out[:, 1] = pts_norm[:, 1] * denom_y
        return out

    def _patient_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        A = self.pack.geom.A
        orgn4 = self.pack.geom.orgn4.reshape(3)
        abc = np.linalg.solve(A, (xyz - orgn4).T).T
        col = abc[:, 0]
        row = abc[:, 1]
        slc = abc[:, 2]
        return np.column_stack([row, col, slc])

    def _voxel_to_patient(self, ijk: np.ndarray) -> np.ndarray:
        A = self.pack.geom.A
        orgn4 = self.pack.geom.orgn4.reshape(3)
        ijk = np.asarray(ijk, dtype=np.float64)
        col = ijk[:, 1]
        row = ijk[:, 0]
        slc = ijk[:, 2]
        abc = np.column_stack([col, row, slc])
        return (orgn4[None, :] + abc @ A.T)

    def _overlay_line_on_plot(self, plot: pg.PlotItem, coords: np.ndarray, color: str, label: str):
        if coords.shape[0] != 2:
            return
        x = coords[:, 0]
        y = coords[:, 1]
        plot.plot(x, y, pen=pg.mkPen(color=color, width=2), name=label)

    def show_plane_overlay(self):
        t = int(self.slider.value()) - 1
        t = int(np.clip(t, 0, self.Nt - 1))
        pcmra3d = self.pack.pcmra[:, :, :, t].astype(np.float32)
        mip_xy = np.max(pcmra3d, axis=2)
        mip_xz = np.max(pcmra3d, axis=0).T
        mip_yz = np.max(pcmra3d, axis=1).T

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Plane Overlay (PCMRA MIPs)")
        dlg.resize(1200, 450)
        layout = QtWidgets.QVBoxLayout(dlg)
        grid = pg.GraphicsLayoutWidget()
        layout.addWidget(grid)

        plot_xy = grid.addPlot(row=0, col=0, title="XY (MIP over Z)")
        plot_xz = grid.addPlot(row=0, col=1, title="XZ (MIP over Y)")
        plot_yz = grid.addPlot(row=0, col=2, title="YZ (MIP over X)")

        img_xy = pg.ImageItem(mip_xy)
        img_xz = pg.ImageItem(mip_xz)
        img_yz = pg.ImageItem(mip_yz)
        plot_xy.addItem(img_xy)
        plot_xz.addItem(img_xz)
        plot_yz.addItem(img_yz)

        plot_xy.setAspectLocked(True)
        plot_xz.setAspectLocked(True)
        plot_yz.setAspectLocked(True)

        cine_colors = {"2ch": "r", "3ch": "m", "4ch": "c"}
        cine_keys = [k for k in ("2ch", "3ch", "4ch") if k in self.pack.cine_planes]
        if not cine_keys:
            cine_keys = [self.active_cine_key]

        for cine_key in cine_keys:
            cine_img_raw = self._get_cine_frame_raw(cine_key, t)
            H_raw, W_raw = cine_img_raw.shape
            if self.line_norm[t] is None:
                self.line_norm[t] = self._default_line_norm()
            line_xy = self._norm_to_abs_line(self.line_norm[t], H_raw, W_raw)
            cine_geom = self._get_cine_geom_raw(cine_key)
            patient_xyz = cine_line_to_patient_xyz(line_xy, cine_geom)
            vox = self._patient_to_voxel(patient_xyz)

            color = cine_colors.get(cine_key, "y")
            self._overlay_line_on_plot(plot_xy, vox[:, [1, 0]], color, cine_key)
            self._overlay_line_on_plot(plot_xz, vox[:, [1, 2]], color, cine_key)
            self._overlay_line_on_plot(plot_yz, vox[:, [0, 2]], color, cine_key)

        dlg.exec()

    def _default_line_norm(self) -> np.ndarray:
        return np.array([[0.45, 0.55], [0.60, 0.45]], dtype=np.float64)

    def on_cine_line_changed_live(self, cine_key: str):
        if self._syncing_cine_line or cine_key != self.active_cine_key:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            return

        img_raw = self._get_cine_frame_raw(cine_key, t)
        H_raw, W_raw = img_raw.shape
        pts_abs = self._roi_points_abs(self.cine_line_rois[cine_key])
        pts_abs = self._canonicalize_line_abs(pts_abs)
        self.line_norm[t] = self._abs_to_norm_line(pts_abs, H_raw, W_raw)

    def on_cine_line_changed_finished(self, cine_key: str):
        if self._syncing_cine_line or cine_key != self.active_cine_key:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            return

        img_raw = self._get_cine_frame_raw(cine_key, t)
        H_raw, W_raw = img_raw.shape
        pts_abs = self._roi_points_abs(self.cine_line_rois[cine_key])
        pts_abs = self._canonicalize_line_abs(pts_abs)
        self.line_norm[t] = self._abs_to_norm_line(pts_abs, H_raw, W_raw)

        Ipcm, Ivelmag, _, _, extras = self.reslice_for_phase(t)
        self._set_image_keep_zoom("pcmra", Ipcm, auto_levels=False)
        self._set_image_keep_zoom("vel", self._display_image(Ivelmag, extras), auto_levels=False)

        self.compute_current(update_only=True)

    def _update_cine_roi_visibility(self):
        for k, roi in self.cine_line_rois.items():
            roi.setVisible(k == self.active_cine_key)

    # ============================
    # Phase update
    # ============================
    def on_phase_changed(self, v: int):
        self._remember_current_levels(self._current_display_mode)
        self.set_phase(v - 1)

    def set_phase(self, t: int):
        t = int(np.clip(t, 0, self.Nt - 1))
        if self._cur_phase == t:
            return
        self._cur_phase = t
        self.lbl_phase.setText(f"Phase: {t + 1}")
        self._apply_display_colormap()
        self._sync_level_controls_from_state()
        self._apply_levels_if_set()

        cine_auto_levels = self._cine_auto_once
        for k, view in self.cine_views.items():
            img = self._get_cine_frame(k, t)
            self._set_image_keep_zoom(f"cine:{k}", img, auto_levels=cine_auto_levels)

        if self.line_norm[t] is None:
            if t > 0 and self.line_norm[t - 1] is not None:
                self.line_norm[t] = np.array(self.line_norm[t - 1], dtype=np.float64).copy()
            else:
                self.line_norm[t] = self._default_line_norm()

        self._syncing_cine_line = True
        try:
            img_raw = self._get_cine_frame_raw(self.active_cine_key, t)
            H_raw, W_raw = img_raw.shape
            pts_abs = self._norm_to_abs_line(self.line_norm[t], H_raw, W_raw)
            self._apply_line_abs_to_roi(self.cine_line_rois[self.active_cine_key], pts_abs, H_raw, W_raw)
        finally:
            self._syncing_cine_line = False

        Ipcm, Ivelmag, _, _, extras = self.reslice_for_phase(t)
        vel_auto_levels = self._vel_auto_levels_enabled()
        pcmra_auto_levels = self._pcmra_auto_once
        self._set_image_keep_zoom("pcmra", Ipcm, auto_levels=pcmra_auto_levels)
        self._set_image_keep_zoom(
            "vel", self._display_image(Ivelmag, extras), auto_levels=vel_auto_levels
        )
        if vel_auto_levels:
            self._capture_auto_levels_once()
        if pcmra_auto_levels:
            self._capture_pcmra_auto_once()
        if cine_auto_levels:
            self._capture_cine_auto_once()

        if self.roi_state[t] is None:
            self.roi_state[t] = self.default_poly_roi_state_from_image(Ivelmag)

        self.ensure_poly_rois()
        self.apply_roi_state_both(self.roi_state[t])
        self.update_spline_overlay(t)
        self.update_metric_labels(t)
        self._update_roi_lock_ui()
        self.set_poly_editable(self.edit_mode and not self._is_roi_locked(t))
        self._update_line_editable(t)
        self._update_lock_label_visibility()
        self._update_lock_label_positions()
        self.plot.set_phase_indicator(t + 1)

    # ============================
    # Reslice
    # ============================
    def _get_active_line_abs_raw(self, t: int) -> np.ndarray:
        img_raw = self._get_cine_frame_raw(self.active_cine_key, t)
        H_raw, W_raw = img_raw.shape
        if self.line_norm[t] is None:
            self.line_norm[t] = self._default_line_norm()
        return self._norm_to_abs_line(self.line_norm[t], H_raw, W_raw)

    def reslice_for_phase(self, t: int):
        line_xy = self._get_active_line_abs_raw(t)
        pcmra3d = self.pack.pcmra[:, :, :, t].astype(np.float32)
        vel5d = self.pack.vel.astype(np.float32)
        cine_geom = self._get_cine_geom_raw(self.active_cine_key)

        extra_scalars: Dict[str, np.ndarray] = {}
        if self.pack.ke is not None and self.pack.ke.ndim == 4:
            extra_scalars["ke"] = self.pack.ke[:, :, :, t].astype(np.float32)
        if self.pack.vortmag is not None and self.pack.vortmag.ndim == 4:
            extra_scalars["vortmag"] = self.pack.vortmag[:, :, :, t].astype(np.float32)

        Ipcm, Ivelmag, Vn, spmm, extras = reslice_plane_fixedN(
            pcmra3d=pcmra3d,
            vel5d=vel5d,
            t=t,
            vol_geom=self.pack.geom,
            cine_geom=cine_geom,
            line_xy=line_xy,
            Npix=self.Npix,
            extra_scalars=extra_scalars,
        )
        Ipcm = Ipcm.T
        Ivelmag = Ivelmag.T
        Vn = Vn.T
        extras = {key: value.T for key, value in extras.items()}
        return Ipcm, Ivelmag, Vn, spmm, extras

    def _display_image(self, Ivelmag: np.ndarray, extras: Dict[str, np.ndarray]) -> np.ndarray:
        mode = self.display_selector.currentText()
        if mode == "Kinetic energy":
            return extras.get("ke", np.zeros_like(Ivelmag))
        if mode == "Vorticity":
            return extras.get("vortmag", np.zeros_like(Ivelmag))
        return Ivelmag

    def on_display_changed(self, _text: str):
        if self._cur_phase is None:
            return
        self._remember_current_levels(self._current_display_mode)
        self._current_display_mode = self.display_selector.currentText()
        Ipcm, Ivelmag, _, _, extras = self.reslice_for_phase(self._cur_phase)
        self._apply_display_colormap()
        self._set_image_keep_zoom(
            "vel", self._display_image(Ivelmag, extras), auto_levels=self._vel_auto_levels_enabled()
        )
        self._capture_auto_levels_once()
        self._sync_level_controls_from_state()
        self._apply_levels_if_set()

    def _apply_display_colormap(self):
        mode = self.display_selector.currentText()
        cmap = self._display_colormaps.get(mode)
        if cmap is not None:
            try:
                self.vel_view.setColorMap(cmap)
            except Exception:
                pass

    def _sync_level_controls_from_state(self):
        mode = self.display_selector.currentText()
        vmin, vmax = self._display_levels.get(mode, (None, None))
        self.level_min.blockSignals(True)
        self.level_max.blockSignals(True)
        try:
            if vmin is None or vmax is None:
                self.level_min.setValue(0.0)
                self.level_max.setValue(1.0)
            else:
                self.level_min.setValue(float(vmin))
                self.level_max.setValue(float(vmax))
        finally:
            self.level_min.blockSignals(False)
            self.level_max.blockSignals(False)

        pcmra_vmin, pcmra_vmax = self._pcmra_levels
        self.pcmra_min.blockSignals(True)
        self.pcmra_max.blockSignals(True)
        try:
            if pcmra_vmin is None or pcmra_vmax is None:
                self.pcmra_min.setValue(0.0)
                self.pcmra_max.setValue(1.0)
            else:
                self.pcmra_min.setValue(float(pcmra_vmin))
                self.pcmra_max.setValue(float(pcmra_vmax))
        finally:
            self.pcmra_min.blockSignals(False)
            self.pcmra_max.blockSignals(False)

        cine_vmin, cine_vmax = self._cine_levels
        self.cine_min.blockSignals(True)
        self.cine_max.blockSignals(True)
        try:
            if cine_vmin is None or cine_vmax is None:
                self.cine_min.setValue(0.0)
                self.cine_max.setValue(1.0)
            else:
                self.cine_min.setValue(float(cine_vmin))
                self.cine_max.setValue(float(cine_vmax))
        finally:
            self.cine_min.blockSignals(False)
            self.cine_max.blockSignals(False)

    def _apply_levels_if_set(self):
        mode = self.display_selector.currentText()
        vmin, vmax = self._display_levels.get(mode, (None, None))
        if vmin is None or vmax is None:
            return
        try:
            self.vel_view.setLevels(vmin, vmax)
        except Exception:
            pass

    def _remember_current_levels(self, mode: str):
        try:
            levels = self.vel_view.getLevels()
        except Exception:
            return
        if levels is None or len(levels) != 2:
            return
        vmin, vmax = levels
        if vmin is None or vmax is None:
            return
        self._display_levels[mode] = (float(vmin), float(vmax))

    def _vel_auto_levels_enabled(self) -> bool:
        mode = self.display_selector.currentText()
        return bool(self._vel_auto_once.get(mode, False))

    def _capture_auto_levels_once(self):
        mode = self.display_selector.currentText()
        if not self._vel_auto_levels_enabled():
            return
        try:
            levels = self.vel_view.getLevels()
        except Exception:
            return
        if levels is None or len(levels) != 2:
            return
        vmin, vmax = levels
        if vmin is None or vmax is None:
            return
        self._display_levels[mode] = (float(vmin), float(vmax))
        self._vel_auto_once[mode] = False

    def _capture_pcmra_auto_once(self):
        if not self._pcmra_auto_once:
            return
        try:
            levels = self.pcmra_view.getLevels()
        except Exception:
            return
        if levels is None or len(levels) != 2:
            return
        vmin, vmax = levels
        if vmin is None or vmax is None:
            return
        self._pcmra_levels = (float(vmin), float(vmax))
        self._pcmra_auto_once = False

    def _capture_cine_auto_once(self):
        if not self._cine_auto_once:
            return
        view = self.cine_views.get(self.active_cine_key)
        if view is None:
            return
        try:
            levels = view.getLevels()
        except Exception:
            return
        if levels is None or len(levels) != 2:
            return
        vmin, vmax = levels
        if vmin is None or vmax is None:
            return
        self._cine_levels = (float(vmin), float(vmax))
        self._cine_auto_once = False

    def on_level_spin_changed(self, _value: float):
        mode = self.display_selector.currentText()
        self._display_levels[mode] = (float(self.level_min.value()), float(self.level_max.value()))
        self._vel_auto_once[mode] = False

    def apply_level_range(self):
        mode = self.display_selector.currentText()
        vmin = float(self.level_min.value())
        vmax = float(self.level_max.value())
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self._display_levels[mode] = (vmin, vmax)
        self._vel_auto_once[mode] = False
        try:
            self.vel_view.setLevels(vmin, vmax)
        except Exception:
            pass

    def enable_auto_levels(self):
        mode = self.display_selector.currentText()
        self._display_levels[mode] = (None, None)
        self._vel_auto_once[mode] = True
        if self._cur_phase is not None:
            cur = self._cur_phase
            self._cur_phase = None
            self.set_phase(cur)

    def apply_pcmra_levels(self):
        vmin = float(self.pcmra_min.value())
        vmax = float(self.pcmra_max.value())
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self._pcmra_levels = (vmin, vmax)
        self._pcmra_auto_once = False
        try:
            self.pcmra_view.setLevels(vmin, vmax)
        except Exception:
            pass

    def enable_pcmra_auto(self):
        self._pcmra_levels = (None, None)
        self._pcmra_auto_once = True
        if self._cur_phase is not None:
            cur = self._cur_phase
            self._cur_phase = None
            self.set_phase(cur)

    def apply_cine_levels(self):
        vmin = float(self.cine_min.value())
        vmax = float(self.cine_max.value())
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self._cine_levels = (vmin, vmax)
        self._cine_auto_once = False
        for view in self.cine_views.values():
            try:
                view.setLevels(vmin, vmax)
            except Exception:
                pass

    def enable_cine_auto(self):
        self._cine_levels = (None, None)
        self._cine_auto_once = True
        if self._cur_phase is not None:
            cur = self._cur_phase
            self._cur_phase = None
            self.set_phase(cur)

    def convert_to_stl(self):
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return
        if self.active_cine_key not in self.pack.cine_planes:
            return
        if self.line_norm[t] is None:
            self.line_norm[t] = self._default_line_norm()
        line_xy = self._get_active_line_abs_raw(t)
        roi_abs = self._roi_abs_points_from_item()
        cine_geom = self._get_cine_geom_raw(self.active_cine_key)
        vol_shape = self.pack.vel.shape[:3]

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save STL",
            os.path.join(self.work_folder, f"mvtrack_phase_{t + 1}.stl"),
            "STL Files (*.stl)",
        )
        if not out_path:
            return
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        convert_plane_to_stl(
            out_path=out_path,
            vol_geom=self.pack.geom,
            cine_geom=cine_geom,
            line_xy=line_xy,
            roi_abs_pts=roi_abs,
            vol_shape=vol_shape,
            npix=self.Npix,
        )
        if not os.path.exists(out_path):
            self.memo.appendPlainText(f"STL save failed: {out_path}")
            return
        self.memo.appendPlainText(f"STL saved: {out_path}")

    # ============================
    # ROI state
    # ============================
    def default_poly_roi_state(self, shape_hw: Tuple[int, int]) -> dict:
        H, W = shape_hw
        cx, cy = W * 0.5, H * 0.5
        r = min(H, W) * 0.12
        angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)]).astype(np.float64)
        return {"pos": (cx, cy), "points": pts.tolist(), "closed": True}

    def default_poly_roi_state_from_image(self, image: np.ndarray) -> dict:
        H, W = image.shape
        if np.isfinite(image).any():
            iy, ix = np.unravel_index(np.nanargmax(image), image.shape)
            cx, cy = float(ix), float(iy)
        else:
            cx, cy = W * 0.5, H * 0.5
        r = min(H, W) * 0.12
        angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)]).astype(np.float64)
        return {"pos": (cx, cy), "points": pts.tolist(), "closed": True}

    def _set_roi_invisible(self, roi: pg.ROI):
        invisible = pg.mkPen((0, 0, 0, 0))
        try:
            roi.setPen(invisible)
        except Exception:
            pass
        if hasattr(roi, "setHoverPen"):
            try:
                roi.setHoverPen(invisible)
            except Exception:
                pass
        else:
            try:
                roi.hoverPen = invisible
            except Exception:
                pass

    def ensure_poly_rois(self):
        if self.poly_roi_pcm is None:
            self.poly_roi_pcm = pg.PolyLineROI([[0, 0], [10, 0], [10, 10]], closed=True)
            self.pcmra_view.getView().addItem(self.poly_roi_pcm)
            self.poly_roi_pcm.sigRegionChanged.connect(self.on_poly_changed_pcm_live)
            self.poly_roi_pcm.sigRegionChangeFinished.connect(self.on_poly_changed_pcm_finished)
            self._set_roi_invisible(self.poly_roi_pcm)

        if self.poly_roi_vel is None:
            self.poly_roi_vel = pg.PolyLineROI([[0, 0], [10, 0], [10, 10]], closed=True)
            self.vel_view.getView().addItem(self.poly_roi_vel)
            self.poly_roi_vel.sigRegionChanged.connect(self.on_poly_changed_vel_live)
            self.poly_roi_vel.sigRegionChangeFinished.connect(self.on_poly_changed_vel_finished)
            self._set_roi_invisible(self.poly_roi_vel)

        if self.spline_curve_pcm is None:
            self.spline_curve_pcm = pg.PlotCurveItem(pen=pg.mkPen("c", width=2))
            self.pcmra_view.getView().addItem(self.spline_curve_pcm)

        if self.spline_curve_vel is None:
            self.spline_curve_vel = pg.PlotCurveItem(pen=pg.mkPen("w", width=2))
            self.vel_view.getView().addItem(self.spline_curve_vel)

        self.set_poly_editable(self.edit_mode)

    def set_poly_editable(self, editable: bool):
        if self.poly_roi_vel is None or self.poly_roi_pcm is None:
            return

        btns = QtCore.Qt.MouseButton.LeftButton if editable else QtCore.Qt.MouseButton.NoButton
        for roi in (self.poly_roi_pcm, self.poly_roi_vel):
            roi.translatable = True
            roi.setAcceptedMouseButtons(btns)
            for h in roi.getHandles():
                if hasattr(h, "setAcceptedMouseButtons"):
                    h.setAcceptedMouseButtons(btns)
                if hasattr(h, "setVisible"):
                    h.setVisible(bool(editable))

    def _update_line_editable(self, t: int):
        if self.active_cine_key not in self.cine_line_rois:
            return
        roi = self.cine_line_rois[self.active_cine_key]
        editable = not self._is_roi_locked(t)
        btns = QtCore.Qt.MouseButton.LeftButton if editable else QtCore.Qt.MouseButton.NoButton
        roi.setAcceptedMouseButtons(btns)
        for h in roi.getHandles():
            if hasattr(h, "setAcceptedMouseButtons"):
                h.setAcceptedMouseButtons(btns)
            if hasattr(h, "setVisible"):
                h.setVisible(bool(editable))

    def _get_roi_state(self, roi: pg.PolyLineROI) -> dict:
        st = roi.getState()
        pos = st.get("pos", (0, 0))
        pts = st.get("points", [])
        closed = bool(st.get("closed", True))
        pos = (float(pos[0]), float(pos[1]))
        pts = np.array(pts, dtype=np.float64).tolist()
        return {"pos": pos, "points": pts, "closed": closed}

    def _apply_roi_state(self, roi: pg.PolyLineROI, state: dict):
        roi.blockSignals(True)
        try:
            roi.setPos(pg.Point(state["pos"][0], state["pos"][1]))
            roi.setPoints(state["points"], closed=bool(state.get("closed", True)))
        finally:
            roi.blockSignals(False)

    def apply_roi_state_both(self, state: dict):
        if self.poly_roi_pcm is None or self.poly_roi_vel is None:
            return
        self._syncing_poly = True
        try:
            self._apply_roi_state(self.poly_roi_pcm, state)
            self._apply_roi_state(self.poly_roi_vel, state)
        finally:
            self._syncing_poly = False

    def get_current_roi_points_absolute(self) -> np.ndarray:
        t = int(self.slider.value()) - 1
        st = self.roi_state[t]
        if st is None:
            st = self.default_poly_roi_state((self.Npix, self.Npix))
            self.roi_state[t] = st
        pos = np.array(st["pos"], dtype=np.float64)
        pts = np.array(st["points"], dtype=np.float64)
        return pts + pos[None, :]

    def update_spline_overlay(self, t: int):
        if self.spline_curve_pcm is None or self.spline_curve_vel is None:
            return
        st = self.roi_state[t]
        if st is None:
            return
        pos = np.array(st["pos"], dtype=np.float64)
        pts = np.array(st["points"], dtype=np.float64)
        abs_pts = pts + pos[None, :]
        spl = closed_spline_xy(abs_pts, n_out=400)
        if spl is None or len(spl) < 3:
            return
        spl2 = np.vstack([spl, spl[0]])
        x = spl2[:, 0]
        y = spl2[:, 1]
        self.spline_curve_pcm.setData(x, y)
        self.spline_curve_vel.setData(x, y)

    def on_poly_changed_pcm_live(self):
        if self._syncing_poly:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            return
        self.roi_state[t] = self._get_roi_state(self.poly_roi_pcm)
        self.update_spline_overlay(t)

    def on_poly_changed_pcm_finished(self):
        if self._syncing_poly:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            self.apply_roi_state_both(self.roi_state[t])
            self.update_spline_overlay(t)
            return
        state = self._get_roi_state(self.poly_roi_pcm)
        self.roi_state[t] = state
        self.apply_roi_state_both(state)
        self.update_spline_overlay(t)
        self.compute_current(update_only=True)

    def on_poly_changed_vel_live(self):
        if self._syncing_poly:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            return
        self.roi_state[t] = self._get_roi_state(self.poly_roi_vel)
        self.update_spline_overlay(t)

    def on_poly_changed_vel_finished(self):
        if self._syncing_poly:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            self.apply_roi_state_both(self.roi_state[t])
            self.update_spline_overlay(t)
            return
        state = self._get_roi_state(self.poly_roi_vel)
        self.roi_state[t] = state
        self.apply_roi_state_both(state)
        self.update_spline_overlay(t)
        self.compute_current(update_only=True)

    # ============================
    # Metrics
    # ============================
    def toggle_edit(self):
        self.edit_mode = not self.edit_mode
        self.btn_edit.setText("Edit ROI: ON" if self.edit_mode else "Edit ROI: OFF")
        t = int(self.slider.value()) - 1
        self.set_poly_editable(self.edit_mode and not self._is_roi_locked(t))

    def _compute_metrics(
        self,
        Ivelmag: np.ndarray,
        Vn: np.ndarray,
        spmm: float,
        Ike: Optional[np.ndarray],
        Ivort: Optional[np.ndarray],
    ):
        abs_pts = self._roi_abs_points_from_item()
        abs_pts = closed_spline_xy(abs_pts, n_out=400)

        # 안전 클립(혹시 spline이 살짝 밖으로 나가도 OK)
        H, W = Ivelmag.shape
        abs_pts[:, 0] = np.clip(abs_pts[:, 0], 0, W - 1)
        abs_pts[:, 1] = np.clip(abs_pts[:, 1], 0, H - 1)

        mask = polygon_mask(Ivelmag.shape, abs_pts)

        if not np.any(mask):
            Q = np.nan
            Vpk = np.nan
            Vmn = np.nan
            KE = np.nan
            VortPk = np.nan
            VortMn = np.nan
        else:
            vvals = Ivelmag[mask]
            Vpk = float(np.nanmax(vvals)) if vvals.size else np.nan
            Vmn = float(np.nanmean(vvals)) if vvals.size else np.nan

            dA_m2 = (spmm * 1e-3) ** 2
            Q_m3s = float(np.nansum(Vn[mask]) * dA_m2)
            Q = Q_m3s * 1e6

            if Ike is not None:
                KE = float(np.nansum(Ike[mask]) * self._voxel_volume_m3 * 1e6)
            else:
                KE = np.nan

            if Ivort is not None:
                vort_vals = Ivort[mask]
                VortPk = float(np.nanmax(vort_vals)) if vort_vals.size else np.nan
                VortMn = float(np.nanmean(vort_vals)) if vort_vals.size else np.nan
            else:
                VortPk = np.nan
                VortMn = np.nan

        return Q, Vpk, Vmn, KE, VortPk, VortMn

    def _sync_roi_state_for_current_phase(self) -> None:
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return
        if self._is_roi_locked(t):
            return
        if self.poly_roi_vel is not None:
            self.roi_state[t] = self._get_roi_state(self.poly_roi_vel)
        elif self.poly_roi_pcm is not None:
            self.roi_state[t] = self._get_roi_state(self.poly_roi_pcm)

    def compute_current(self, update_only: bool = False):
        t = int(self.slider.value()) - 1
        self._sync_roi_state_for_current_phase()
        Ipcm, Ivelmag, Vn, spmm, extras = self.reslice_for_phase(t)
        Ike = extras.get("ke")
        Ivort = extras.get("vortmag")

        Q, Vpk, Vmn, KE, VortPk, VortMn = self._compute_metrics(Ivelmag, Vn, spmm, Ike, Ivort)

        self.metrics_Q[t] = Q
        self.metrics_Vpk[t] = Vpk
        self.metrics_Vmn[t] = Vmn
        self.metrics_KE[t] = KE
        self.metrics_VortPk[t] = VortPk
        self.metrics_VortMn[t] = VortMn

        self.update_metric_labels(t)

        if not update_only:
            self.update_plot_for_selection()

    def compute_all(self):
        for t in range(self.Nt):
            Ipcm, Ivelmag, Vn, spmm, extras = self.reslice_for_phase(t)
            Ike = extras.get("ke")
            Ivort = extras.get("vortmag")

            if self.roi_state[t] is None:
                self.roi_state[t] = self.default_poly_roi_state_from_image(Ivelmag)

            abs_pts = self._abs_pts_from_state_safe(self.roi_state[t], Ivelmag.shape)
            abs_pts = closed_spline_xy(abs_pts, n_out=400)
            mask = polygon_mask(Ivelmag.shape, abs_pts)

            if not np.any(mask):
                self.metrics_Q[t] = np.nan
                self.metrics_Vpk[t] = np.nan
                self.metrics_Vmn[t] = np.nan
                self.metrics_KE[t] = np.nan
                self.metrics_VortPk[t] = np.nan
                self.metrics_VortMn[t] = np.nan
                continue

            vvals = Ivelmag[mask]
            self.metrics_Vpk[t] = float(np.nanmax(vvals)) if vvals.size else np.nan
            self.metrics_Vmn[t] = float(np.nanmean(vvals)) if vvals.size else np.nan

            dA_m2 = (spmm * 1e-3) ** 2
            Q_m3s = float(np.nansum(Vn[mask]) * dA_m2)
            self.metrics_Q[t] = Q_m3s * 1e6

            if Ike is not None:
                self.metrics_KE[t] = float(np.nansum(Ike[mask]) * self._voxel_volume_m3 * 1e6)
            else:
                self.metrics_KE[t] = np.nan

            if Ivort is not None:
                vort_vals = Ivort[mask]
                self.metrics_VortPk[t] = float(np.nanmax(vort_vals)) if vort_vals.size else np.nan
                self.metrics_VortMn[t] = float(np.nanmean(vort_vals)) if vort_vals.size else np.nan
            else:
                self.metrics_VortPk[t] = np.nan
                self.metrics_VortMn[t] = np.nan

        self.update_plot_for_selection()
        self.update_metric_labels(int(self.slider.value()) - 1)
        self.memo.appendPlainText("Compute all done.")

    def _abs_pts_from_state_safe(self, st: dict, shape_hw: Tuple[int, int]) -> np.ndarray:
        H, W = shape_hw
        pos = np.array(st.get("pos", (0.0, 0.0)), dtype=np.float64)
        pts = np.array(st.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return pts

        cand_local = pts + pos[None, :]
        cand_abs = pts

        def score(P):
            x, y = P[:, 0], P[:, 1]
            return np.mean((x >= 0) & (x < W) & (y >= 0) & (y < H))

        abs_pts = cand_local if score(cand_local) >= score(cand_abs) else cand_abs
        return abs_pts

    # ============================
    # Missing helpers
    # ============================
    def _store_view_range(self, key: str):
        if self._restoring_view or self._updating_image:
            return
        view = self._view_for_key(key)
        self._view_ranges[key] = view.viewRange()

    def _restore_view_range(self, key: str):
        view = self._view_for_key(key)
        rng = self._view_ranges.get(key)
        if rng is None:
            return
        self._restoring_view = True
        try:
            view.setRange(xRange=rng[0], yRange=rng[1], padding=0.0)
        finally:
            self._restoring_view = False

    def _set_image_keep_zoom(self, key: str, image: np.ndarray, auto_levels: bool = False):
        self._updating_image = True
        try:
            view = self._image_view_for_key(key)
            have_range = self._view_ranges.get(key) is not None
            view.setImage(
                image,
                autoLevels=auto_levels,
                autoRange=(not have_range),
                autoHistogramRange=(not have_range),
            )
            if key == "pcmra":
                vmin, vmax = self._pcmra_levels
                if vmin is not None and vmax is not None:
                    try:
                        view.setLevels(vmin, vmax)
                    except Exception:
                        pass
            elif key.startswith("cine:"):
                vmin, vmax = self._cine_levels
                if vmin is not None and vmax is not None:
                    try:
                        view.setLevels(vmin, vmax)
                    except Exception:
                        pass
            elif key == "vel":
                mode = self.display_selector.currentText()
                vmin, vmax = self._display_levels.get(mode, (None, None))
                if vmin is not None and vmax is not None:
                    try:
                        self.vel_view.setLevels(vmin, vmax)
                    except Exception:
                        pass
        finally:
            self._updating_image = False
        self._restore_view_range(key)

    def update_metric_labels(self, t: int):
        Q = self.metrics_Q[t]
        Vpk = self.metrics_Vpk[t]
        Vmn = self.metrics_Vmn[t]
        KE = self.metrics_KE[t]
        VortPk = self.metrics_VortPk[t]
        VortMn = self.metrics_VortMn[t]
        self.lbl_Q.setText(f"Flow rate (mL/s): {Q:.3f}" if np.isfinite(Q) else "Flow rate (mL/s): -")
        self.lbl_Vpk.setText(f"Peak velocity (m/s): {Vpk:.3f}" if np.isfinite(Vpk) else "Peak velocity (m/s): -")
        self.lbl_Vmn.setText(f"Mean velocity (m/s): {Vmn:.3f}" if np.isfinite(Vmn) else "Mean velocity (m/s): -")
        self.lbl_KE.setText(f"Kinetic energy (uJ): {KE:.3f}" if np.isfinite(KE) else "Kinetic energy (uJ): -")
        self.lbl_VortPk.setText(
            f"Peak vorticity (1/s): {VortPk:.3f}" if np.isfinite(VortPk) else "Peak vorticity (1/s): -"
        )
        self.lbl_VortMn.setText(
            f"Mean vorticity (1/s): {VortMn:.3f}" if np.isfinite(VortMn) else "Mean vorticity (1/s): -"
        )

    def update_plot_for_selection(self):
        phases = np.arange(1, self.Nt + 1)
        label = self.chart_selector.currentText()
        if label == "Flow rate (mL/s)":
            self.plot.plot_metric(phases, self.metrics_Q, label, "tab:blue")
        elif label == "Peak velocity (m/s)":
            self.plot.plot_metric(phases, self.metrics_Vpk, label, "tab:orange")
        elif label == "Mean velocity (m/s)":
            self.plot.plot_metric(phases, self.metrics_Vmn, label, "tab:green")
        elif label == "Kinetic energy (uJ)":
            self.plot.plot_metric(phases, self.metrics_KE, label, "tab:red")
        elif label == "Peak vorticity (1/s)":
            self.plot.plot_metric(phases, self.metrics_VortPk, label, "tab:purple")
        elif label == "Mean vorticity (1/s)":
            self.plot.plot_metric(phases, self.metrics_VortMn, label, "tab:brown")
        self.plot.set_phase_indicator(int(self.slider.value()))

    def on_plot_phase_selected(self, phase: int):
        if phase is None:
            return
        phase = int(np.clip(phase, 1, self.Nt))
        if phase != self.slider.value():
            self.slider.setValue(phase)

    def _is_roi_locked(self, t: int) -> bool:
        if t < 0 or t >= self.Nt:
            return False
        return bool(self.roi_locked[t])

    def _update_roi_lock_ui(self):
        t = int(self.slider.value()) - 1
        locked = self._is_roi_locked(t)
        self.btn_roi_lock.setText("Lock ROI: ON" if locked else "Lock ROI: OFF")
        self._update_lock_label_visibility()

    def toggle_roi_lock(self):
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return
        self.roi_locked[t] = not self.roi_locked[t]
        if self.roi_locked[t]:
            self.roi_state[t] = self._get_roi_state(self.poly_roi_pcm)
        self._update_roi_lock_ui()
        self.set_poly_editable(self.edit_mode and not self._is_roi_locked(t))
        self._update_line_editable(t)
        self._update_lock_label_visibility()
        self._update_lock_label_positions()

    def copy_roi_state(self):
        t = int(self.slider.value()) - 1
        st = self.roi_state[t]
        if st is None:
            return
        self._roi_clipboard = json.loads(json.dumps(st))
        self.memo.appendPlainText("Copied ROI state.")

    def paste_roi_state(self):
        if self._roi_clipboard is None:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            self.memo.appendPlainText("ROI is locked for this phase.")
            return
        state = json.loads(json.dumps(self._roi_clipboard))
        self.roi_state[t] = state
        self.apply_roi_state_both(state)
        self.update_spline_overlay(t)
        self.compute_current(update_only=True)
        self.memo.appendPlainText("Pasted ROI state.")

    def copy_roi_forward(self):
        t = int(self.slider.value()) - 1
        st = self.roi_state[t]
        if st is None:
            return
        for tt in range(t + 1, self.Nt):
            if not self._is_roi_locked(tt):
                self.roi_state[tt] = json.loads(json.dumps(st))
        self.memo.appendPlainText("Copied ROI state forward.")

    def _update_lock_label_visibility(self):
        t = int(self.slider.value()) - 1
        locked = self._is_roi_locked(t)
        if self.lock_label_pcm is not None:
            self.lock_label_pcm.setVisible(locked)
        if self.lock_label_vel is not None:
            self.lock_label_vel.setVisible(locked)

    def _update_lock_label_positions(self):
        self._position_lock_label(self.pcmra_view, self.lock_label_pcm)
        self._position_lock_label(self.vel_view, self.lock_label_vel)

    def _position_lock_label(self, view: pg.ImageView, label: Optional[pg.TextItem]):
        if label is None or not label.isVisible():
            return
        try:
            (xr, yr) = view.getView().viewRange()
        except Exception:
            return
        pad_x = (xr[1] - xr[0]) * 0.02
        pad_y = (yr[1] - yr[0]) * 0.02
        label.setPos(xr[0] + pad_x, yr[1] - pad_y)

    def _view_for_key(self, key: str) -> pg.ViewBox:
        if key.startswith("cine:"):
            cine_key = key.split(":", 1)[1]
            return self.cine_views[cine_key].getView()
        if key == "pcmra":
            return self.pcmra_view.getView()
        if key == "vel":
            return self.vel_view.getView()
        raise KeyError(f"Unknown view key: {key}")

    def _image_view_for_key(self, key: str) -> pg.ImageView:
        if key.startswith("cine:"):
            cine_key = key.split(":", 1)[1]
            return self.cine_views[cine_key]
        if key == "pcmra":
            return self.pcmra_view
        if key == "vel":
            return self.vel_view
        raise KeyError(f"Unknown image view key: {key}")

    def on_active_cine_changed(self, cine_key: str):
        if cine_key not in self.pack.cine_planes:
            return
        self.active_cine_key = cine_key
        self._update_cine_roi_visibility()
        if self.cine_selector.currentText() != cine_key:
            self.cine_selector.blockSignals(True)
            try:
                self.cine_selector.setCurrentText(cine_key)
            finally:
                self.cine_selector.blockSignals(False)
        if self._cur_phase is not None:
            self.set_phase(self._cur_phase)

    def _sync_current_phase_state(self):
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return
        if self.poly_roi_pcm is not None:
            self.roi_state[t] = self._get_roi_state(self.poly_roi_pcm)
        if self.active_cine_key in self.cine_line_rois:
            img = self._get_cine_frame(self.active_cine_key, t)
            H, W = img.shape
            pts_abs = self._roi_points_abs(self.cine_line_rois[self.active_cine_key])
            pts_abs = self._canonicalize_line_abs(pts_abs)
            self.line_norm[t] = self._abs_to_norm_line(pts_abs, H, W)

    def save_to_mvtrack_h5(self):
        self._sync_current_phase_state()
        default_name = os.path.basename(self.tracking_path) if self.tracking_path else "MVtrack.h5"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save MVtrack",
            "Filename (MVtrack_*.h5):",
            text=default_name,
        )
        if not ok or not name:
            return
        if not name.lower().endswith(".h5"):
            name = f"{name}.h5"
        if not name.lower().startswith("mvtrack"):
            name = f"MVtrack_{name}"
        out_path = os.path.join(self.work_folder, name)
        save_tracking_state_h5(
            path=out_path,
            line_norm=self.line_norm,
            roi_state=self.roi_state,
            roi_locked=self.roi_locked,
            metrics_Q=self.metrics_Q,
            metrics_Vpk=self.metrics_Vpk,
            metrics_Vmn=self.metrics_Vmn,
            metrics_KE=self.metrics_KE,
            metrics_VortPk=self.metrics_VortPk,
            metrics_VortMn=self.metrics_VortMn,
            active_cine_key=self.active_cine_key,
            display_levels=self._display_levels,
            vel_auto_once=self._vel_auto_once,
            cine_levels=self._cine_levels,
            cine_auto_once=self._cine_auto_once,
            pcmra_levels=self._pcmra_levels,
            pcmra_auto_once=self._pcmra_auto_once,
            cine_flip=(
                self.chk_cine_flip_x.isChecked(),
                self.chk_cine_flip_y.isChecked(),
                self.chk_cine_flip_z.isChecked(),
            ),
            cine_swap=self.cine_swap_selector.currentText(),
        )
        self.tracking_path = out_path
        self.memo.appendPlainText(f"Saved tracking state to: {out_path}")

    def copy_current_to_clipboard(self):
        lines = [
            "Phase\tFlow rate (mL/s)\tPeak velocity (m/s)\tMean velocity (m/s)\tKinetic energy (uJ)\tPeak vorticity (1/s)\tMean vorticity (1/s)"
        ]
        for t in range(self.Nt):
            Q = self.metrics_Q[t]
            Vpk = self.metrics_Vpk[t]
            Vmn = self.metrics_Vmn[t]
            KE = self.metrics_KE[t]
            VortPk = self.metrics_VortPk[t]
            VortMn = self.metrics_VortMn[t]
            q_text = f"{Q:.6f}" if np.isfinite(Q) else ""
            vpk_text = f"{Vpk:.6f}" if np.isfinite(Vpk) else ""
            vmn_text = f"{Vmn:.6f}" if np.isfinite(Vmn) else ""
            ke_text = f"{KE:.6f}" if np.isfinite(KE) else ""
            vortpk_text = f"{VortPk:.6f}" if np.isfinite(VortPk) else ""
            vortmn_text = f"{VortMn:.6f}" if np.isfinite(VortMn) else ""
            lines.append(
                f"{t + 1}\t{q_text}\t{vpk_text}\t{vmn_text}\t{ke_text}\t{vortpk_text}\t{vortmn_text}"
            )
        text = "\n".join(lines)
        QtWidgets.QApplication.clipboard().setText(text)
        self.memo.appendPlainText("Copied all phase metrics to clipboard.")

    def _canonicalize_line_abs(self, pts_abs: np.ndarray) -> np.ndarray:
        a = np.asarray(pts_abs[0], dtype=np.float64)
        b = np.asarray(pts_abs[1], dtype=np.float64)
        if (b[0] < a[0]) or (abs(b[0] - a[0]) < 1e-6 and b[1] < a[1]):
            return np.vstack([b, a])
        return np.vstack([a, b])

    def _roi_abs_points_from_item(self) -> np.ndarray:
        """
        현재 화면(vel ROI)에서 poly ROI의 꼭지점을 '절대 좌표'로 가져옴.
        pos/points 해석 꼬임을 피하기 위해 item을 신뢰.
        """
        roi = self.poly_roi_vel if self.poly_roi_vel is not None else self.poly_roi_pcm
        if roi is None:
            return self.get_current_roi_points_absolute()

        st = roi.getState()
        pos = np.array(st.get("pos", (0.0, 0.0)), dtype=np.float64)
        pts = np.array(st.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return self.get_current_roi_points_absolute()

        # 여기서 핵심: pts가 local인지 absolute인지 애매하면, 둘 다 만들어서
        # "이미지 안에 더 잘 들어오는 쪽"을 선택
        H, W = self.Npix, self.Npix

        cand_local = pts + pos[None, :]
        cand_abs = pts

        def in_bounds_score(P):
            x = P[:, 0]
            y = P[:, 1]
            return np.mean((x >= 0) & (x < W) & (y >= 0) & (y < H))

        score_local = in_bounds_score(cand_local)
        score_abs = in_bounds_score(cand_abs)

        abs_pts = cand_local if score_local >= score_abs else cand_abs
        return abs_pts
