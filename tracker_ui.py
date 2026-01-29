import importlib
import importlib.util
import json
import os
import time
from typing import Dict, Tuple, Optional, List

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from scipy.ndimage import map_coordinates

_imageio_spec = importlib.util.find_spec("imageio.v2")
imageio = importlib.import_module("imageio.v2") if _imageio_spec else None
from pcmra_medsam2_refine import (
    get_medsam2_settings,
    mask_to_polygon_points,
    polygon_points_to_roi_state,
    run_medsam2_subprocess,
)
from cinema_subprocess import get_cinema_settings, run_cinema_subprocess
from geometry import (
    apply_axis_transform,
    reslice_plane_fixedN,
    cine_line_to_patient_xyz,
    cine_display_mapping,
    make_plane_from_cine_line,
    auto_fov_from_line,
    transform_vector_components,
)
from stl_conversion import (
    write_stl_from_patient_contour,
    write_stl_from_patient_contour_extruded,
)
from mvpack_io import MVPack, CineGeom, load_mvpack_h5
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

        # ------------------------------------------------------------------
        # FIX: normalize pcmra/vel axis to (row, col, slice, ...)
        # If pack was saved as (col, row, slice, ...), transpose once here
        # so you do NOT need to use YXZ swap later.
        # ------------------------------------------------------------------
        try:
            # pcmra: (X, Y, Z, T) -> (Y, X, Z, T)
            if pack.pcmra.ndim == 4:
                pack.pcmra = np.transpose(pack.pcmra, (1, 0, 2, 3))

            # vel: (X, Y, Z, 3, T) -> (Y, X, Z, 3, T)
            if pack.vel is not None and pack.vel.ndim == 5:
                pack.vel = np.transpose(pack.vel, (1, 0, 2, 3, 4))

                # swap velocity components too: (vx, vy, vz) -> (vy, vx, vz)
                pack.vel = pack.vel[:, :, :, [1, 0, 2], :]

            # optional scalar volumes that follow spatial axes
            if getattr(pack, "ke", None) is not None and pack.ke.ndim == 4:
                pack.ke = np.transpose(pack.ke, (1, 0, 2, 3))
            if getattr(pack, "vortmag", None) is not None and pack.vortmag.ndim == 4:
                pack.vortmag = np.transpose(pack.vortmag, (1, 0, 2, 3))

        except Exception:
            pass
            
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
        self._syncing_view = False
        self._roi_clipboard = None
        self._line_clipboard = None
        self._line_angle_clipboard = None
        self.lock_label_pcm = None
        self.lock_label_vel = None
        self._restored_state = False
        self._cine_display_maps: Dict[str, Tuple[Tuple[int, int], np.ndarray, np.ndarray]] = {}
        self._child_windows = []
        self._segment_ref_angle = [None] * self.Nt
        self._segment_anchor_xy = [None] * self.Nt
        self._segment_count = [6] * self.Nt
        self._show_segments = False
        self.brush_mode = False
        self.brush_radius = 3.0
        self.brush_strength = 0.02
        self._negative_point_mode = False
        self._negative_points: List[List[List[float]]] = [[] for _ in range(self.Nt)]
        self._negative_point_marker = None
        self.play_fps = 10.0
        self._play_timer = QtCore.QTimer(self)
        self.line_angle = [0.0] * self.Nt
        self.metrics_seg4 = {}
        self.metrics_seg6 = {}
        self._segment_label_items_pcm = []
        self._segment_label_items_vel = []
        self._line_marker_enabled = False
        self.line_marker_pcm = None
        self.line_marker_vel = None
        self._history_active = None
        self._undo_stack = []
        self._redo_stack = []
        self._restoring_history = False
        self._redo_shortcuts = []

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)

        layout = QtWidgets.QGridLayout(cw)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 0)
        layout.setColumnStretch(0, 1)

        # LEFT: cine selector + multi cine
        self.cine_views: Dict[str, pg.ImageView] = {}
        self.cine_line_rois: Dict[str, pg.LineSegmentROI] = {}

        left_box = QtWidgets.QVBoxLayout()

        self.btn_load_mvpack = QtWidgets.QPushButton("Load mvpack")
        left_box.addWidget(self.btn_load_mvpack)

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

        self.cine_view_checks: Dict[str, QtWidgets.QCheckBox] = {}
        cine_check_row = QtWidgets.QHBoxLayout()
        left_box.addLayout(cine_check_row)

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
            chk = QtWidgets.QCheckBox(k)
            chk.setChecked(True)
            chk.stateChanged.connect(lambda _state, kk=k: self._update_cine_view_visibility(kk))
            self.cine_view_checks[k] = chk
            cine_check_row.addWidget(chk)
        cine_check_row.addStretch(1)

        # RIGHT: pcmra / vel
        self.pcmra_view = pg.ImageView()
        self.pcmra_view.ui.roiBtn.hide()
        self.pcmra_view.ui.menuBtn.hide()
        self.pcmra_view.ui.histogram.setFixedWidth(90)
        self.pcmra_view.ui.histogram.hide()

        pcmra_box = QtWidgets.QVBoxLayout()

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

        pcmra_ctrl = QtWidgets.QGridLayout()
        pcmra_ctrl.setContentsMargins(0, 0, 0, 0)
        pcmra_ctrl.setHorizontalSpacing(4)
        pcmra_ctrl.setVerticalSpacing(4)
        self.btn_roi_copy = QtWidgets.QPushButton("Copy ROI")
        self.btn_roi_paste = QtWidgets.QPushButton("Paste ROI")
        self.btn_roi_forward = QtWidgets.QPushButton("Copy ROI forward")
        self.btn_edit = QtWidgets.QPushButton("Edit ROI: OFF")
        pcmra_ctrl.addWidget(self.btn_edit, 0, 0)
        pcmra_ctrl.addWidget(self.btn_roi_copy, 0, 1)
        pcmra_ctrl.addWidget(self.btn_roi_paste, 0, 2)
        pcmra_ctrl.addWidget(self.btn_roi_forward, 0, 3)

        self.btn_pcmra_gif = QtWidgets.QPushButton("Export PCMRA GIF")
        self.btn_vel_gif = QtWidgets.QPushButton("Export Colormap GIF")

        self.chk_apply_segments = QtWidgets.QCheckBox("Apply segments")
        self.segment_selector = QtWidgets.QComboBox()
        self.segment_selector.addItems(["4 segments", "6 segments"])
        self.chk_segment_labels = QtWidgets.QCheckBox("Show R labels")
        self.btn_brush = QtWidgets.QPushButton("Brush ROI: OFF")
        pcmra_ctrl.addWidget(self.chk_apply_segments, 0, 4)
        pcmra_ctrl.addWidget(self.segment_selector, 0, 5)
        pcmra_ctrl.addWidget(self.chk_segment_labels, 0, 6)
        pcmra_ctrl.addWidget(self.btn_brush, 0, 7)

        self.btn_refine_roi_phase = QtWidgets.QPushButton("Refine ROI (this phase)")
        self.btn_refine_roi_all = QtWidgets.QPushButton("Refine ROI (all phases)")
        self.chk_negative_points = QtWidgets.QCheckBox("Enable negative points")
        self.chk_negative_points.stateChanged.connect(self._on_negative_points_toggle)
        pcmra_ctrl.addWidget(self.chk_negative_points, 1, 0)
        pcmra_ctrl.addWidget(self.btn_refine_roi_phase, 1, 1)
        pcmra_ctrl.addWidget(self.btn_refine_roi_all, 1, 2)
        pcmra_ctrl.setColumnStretch(8, 1)
        self.pcmra_refine_widget = QtWidgets.QWidget()
        self.pcmra_refine_widget.setLayout(pcmra_ctrl)
        self._configure_pcmra_refine_widget()
        self._configure_cinema_inference_widget()
        self.pcmra_refine_widget.setContentsMargins(0, 0, 0, 0)

        self.vel_view = pg.ImageView()
        self.vel_view.ui.roiBtn.hide()
        self.vel_view.ui.menuBtn.hide()
        self.vel_view.ui.histogram.setFixedWidth(90)
        self.vel_view.ui.histogram.hide()

        vel_box = QtWidgets.QVBoxLayout()

        display_row = QtWidgets.QHBoxLayout()
        vel_box.addLayout(display_row)
        display_row.addWidget(QtWidgets.QLabel("Display"))
        self.display_selector = QtWidgets.QComboBox()
        self.display_selector.addItems(["Velocity", "Kinetic energy", "Vorticity"])
        display_row.addWidget(self.display_selector)
        self._current_display_mode = self.display_selector.currentText()

        display_row.addWidget(QtWidgets.QLabel("Min"))
        self.level_min = QtWidgets.QDoubleSpinBox()
        self.level_min.setDecimals(4)
        self.level_min.setRange(-1e9, 1e9)
        display_row.addWidget(self.level_min)
        display_row.addWidget(QtWidgets.QLabel("Max"))
        self.level_max = QtWidgets.QDoubleSpinBox()
        self.level_max.setDecimals(4)
        self.level_max.setRange(-1e9, 1e9)
        display_row.addWidget(self.level_max)
        self.btn_auto_levels = QtWidgets.QPushButton("Auto")
        display_row.addWidget(self.btn_auto_levels)
        self.btn_apply_levels = QtWidgets.QPushButton("Apply")
        display_row.addWidget(self.btn_apply_levels)
        display_row.addStretch(1)

        vel_box.addWidget(self.vel_view, stretch=1)
        self.display_selector.currentTextChanged.connect(self.on_display_changed)

        vel_ctrl = QtWidgets.QGridLayout()
        vel_ctrl.setContentsMargins(0, 0, 0, 0)
        vel_ctrl.setHorizontalSpacing(4)
        vel_ctrl.setVerticalSpacing(4)
        vel_ctrl.setColumnStretch(0, 1)
        vel_box.addLayout(vel_ctrl)

        self.axis_order = "XYZ"
        self.axis_flips = (False, False, False)

        self.axis_order_selector = QtWidgets.QComboBox()
        self.axis_order_selector.addItems(["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
        self.axis_order_selector.setCurrentText(self.axis_order)
        self.chk_axis_flip_x = QtWidgets.QCheckBox("Flip X")
        self.chk_axis_flip_y = QtWidgets.QCheckBox("Flip Y")
        self.chk_axis_flip_z = QtWidgets.QCheckBox("Flip Z")

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
        self.axis_order = "XYZ"
        self.axis_flips = (False, False, False)
        self._pcmra_levels: Tuple[Optional[float], Optional[float]] = (None, None)
        self._cine_levels: Tuple[Optional[float], Optional[float]] = (None, None)
        self._pcmra_auto_once = True
        self._cine_auto_once = True

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_box)
        pcmra_widget = QtWidgets.QWidget()
        pcmra_widget.setLayout(pcmra_box)
        vel_widget = QtWidgets.QWidget()
        vel_widget.setLayout(vel_box)

        top_area = QtWidgets.QWidget()
        top_layout = QtWidgets.QGridLayout(top_area)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setHorizontalSpacing(8)
        top_layout.setVerticalSpacing(4)
        top_layout.addWidget(left_widget, 0, 0)
        top_layout.addWidget(pcmra_widget, 0, 1)
        top_layout.addWidget(vel_widget, 0, 2)
        left_controls_widget = QtWidgets.QWidget()
        left_controls_layout = QtWidgets.QVBoxLayout(left_controls_widget)
        left_controls_layout.setContentsMargins(0, 0, 0, 0)
        left_controls_layout.setSpacing(4)
        top_layout.addWidget(self.pcmra_refine_widget, 1, 1, 1, 2)
        top_layout.addWidget(left_controls_widget, 1, 0)
        top_layout.setColumnStretch(0, 1)
        top_layout.setColumnStretch(1, 1)
        top_layout.setColumnStretch(2, 1)
        top_layout.setRowStretch(0, 1)
        top_layout.setRowStretch(1, 0)
        bottom_right = QtWidgets.QVBoxLayout()

        chart_log_row = QtWidgets.QHBoxLayout()
        bottom_right.addLayout(chart_log_row, stretch=1)

        chart_box = QtWidgets.QVBoxLayout()
        chart_log_row.addLayout(chart_box, stretch=4)

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
        self.chk_flip_flow = QtWidgets.QCheckBox("Flip flow sign")
        chart_row.addWidget(self.chk_flip_flow)
        self.chk_plot_segments = QtWidgets.QCheckBox("Plot segments")
        chart_row.addWidget(self.chk_plot_segments)
        chart_row.addStretch(1)
        chart_box.addLayout(chart_row)

        self.plot = PlotCanvas()
        chart_box.addWidget(self.plot, stretch=1)
        self.chart_selector.currentTextChanged.connect(self.update_plot_for_selection)
        self.plot.set_phase_callback(self.on_plot_phase_selected)

        log_box = QtWidgets.QVBoxLayout()
        chart_log_row.addLayout(log_box, stretch=1)
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

        self.lbl_segments = QtWidgets.QLabel("Segments: -")

        self.lbl_segments.setVisible(False)
        for w in [
            self.lbl_phase,
            self.lbl_Q,
            self.lbl_Vpk,
            self.lbl_Vmn,
            self.lbl_KE,
            self.lbl_VortPk,
            self.lbl_VortMn,
        ]:
            stats_row.addWidget(w)
        stats_row.addStretch(1)

        btn_row = QtWidgets.QHBoxLayout()
        bottom_right.addLayout(btn_row)

        self.btn_compute = QtWidgets.QPushButton("Compute current")
        self.btn_all = QtWidgets.QPushButton("Compute all")
        self.btn_copy = QtWidgets.QPushButton("Copy data")
        self.btn_copy_regional = QtWidgets.QPushButton("Copy regional data")
        self.copy_regional_selector = QtWidgets.QComboBox()
        self.copy_regional_selector.addItems(["Current chart", "All metrics"])
        self.btn_save = QtWidgets.QPushButton("Save to MVtrack.h5")
        self.btn_convert_stl = QtWidgets.QPushButton("Convert STL")
        self.btn_cine_gif = QtWidgets.QPushButton("Export Cine GIF")

        btn_row.addWidget(self.btn_compute)
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_copy_regional)
        btn_row.addWidget(self.copy_regional_selector)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_convert_stl)
        btn_row.addWidget(self.btn_cine_gif)
        btn_row.addWidget(self.btn_pcmra_gif)
        btn_row.addWidget(self.btn_vel_gif)

        bottom_widget = QtWidgets.QWidget()
        bottom_widget.setLayout(bottom_right)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(top_area)
        splitter.addWidget(bottom_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 0, 0, 2, 1)
        QtCore.QTimer.singleShot(
            0, lambda: splitter.setSizes([int(self.height() * 0.7), int(self.height() * 0.3)])
        )

        self.btn_compute.clicked.connect(self.compute_current)
        self.btn_all.clicked.connect(self.compute_all)
        self.btn_edit.clicked.connect(self.toggle_edit)
        self.btn_copy.clicked.connect(self.copy_data_to_clipboard)
        self.btn_copy_regional.clicked.connect(self.copy_regional_data_to_clipboard)
        self.btn_roi_copy.clicked.connect(self.copy_roi_state)
        self.btn_roi_paste.clicked.connect(self.paste_roi_state)
        self.btn_roi_forward.clicked.connect(self.copy_roi_forward)
        self.btn_refine_roi_phase.clicked.connect(self.refine_roi_pcmra_phase)
        self.btn_refine_roi_all.clicked.connect(self.refine_roi_pcmra_all_phases)
        self.btn_save.clicked.connect(self.save_to_mvtrack_h5)
        self.btn_convert_stl.clicked.connect(self.convert_to_stl)
        self.btn_cine_gif.clicked.connect(self.export_cine_gif)
        self.btn_brush.clicked.connect(self.toggle_brush_mode)
        self.btn_pcmra_gif.clicked.connect(self.export_pcmra_gif)
        self.btn_vel_gif.clicked.connect(self.export_vel_gif)
        self.chk_plot_segments.stateChanged.connect(self.update_plot_for_selection)
        self.chk_flip_flow.stateChanged.connect(self.update_plot_for_selection)
        self.chk_apply_segments.stateChanged.connect(self.toggle_segments_visibility)
        self.segment_selector.currentTextChanged.connect(self.toggle_segments_visibility)
        self.chk_segment_labels.stateChanged.connect(self._on_segment_labels_toggle)
        self.btn_apply_levels.clicked.connect(self.apply_level_range)
        self.btn_auto_levels.clicked.connect(self.enable_auto_levels)
        self.btn_pcmra_apply.clicked.connect(self.apply_pcmra_levels)
        self.btn_pcmra_auto.clicked.connect(self.enable_pcmra_auto)
        self.btn_cine_apply_levels.clicked.connect(self.apply_cine_levels)
        self.btn_cine_auto_levels.clicked.connect(self.enable_cine_auto)
        self.level_min.valueChanged.connect(self.on_level_spin_changed)
        self.level_max.valueChanged.connect(self.on_level_spin_changed)
        self.axis_order_selector.currentTextChanged.connect(self._on_axis_transform_changed)
        self.chk_axis_flip_x.stateChanged.connect(self._on_axis_transform_changed)
        self.chk_axis_flip_y.stateChanged.connect(self._on_axis_transform_changed)
        self.chk_axis_flip_z.stateChanged.connect(self._on_axis_transform_changed)

        copy_action = QtGui.QAction(self)
        copy_action.setShortcut(QtGui.QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_roi_state)
        self.addAction(copy_action)

        paste_action = QtGui.QAction(self)
        paste_action.setShortcut(QtGui.QKeySequence.Paste)
        paste_action.triggered.connect(self.paste_roi_state)
        self.addAction(paste_action)

        line_copy_action = QtGui.QAction(self)
        line_copy_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+C"))
        line_copy_action.triggered.connect(self.copy_line_state)
        self.addAction(line_copy_action)

        line_paste_action = QtGui.QAction(self)
        line_paste_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+V"))
        line_paste_action.triggered.connect(self.paste_line_state)
        self.addAction(line_paste_action)

        undo_action = QtGui.QAction(self)
        undo_action.setShortcut(QtGui.QKeySequence.Undo)
        undo_action.triggered.connect(self.undo_last_action)
        self.addAction(undo_action)

        redo_action = QtGui.QAction(self)
        redo_action.setShortcuts([QtGui.QKeySequence.Redo, QtGui.QKeySequence("Ctrl+Y")])
        redo_action.triggered.connect(self.redo_last_action)
        self.addAction(redo_action)

        play_action = QtGui.QAction(self)
        play_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        play_action.setShortcutContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        play_action.triggered.connect(lambda: self.toggle_playback(not self.btn_play.isChecked()))
        self.addAction(play_action)

        self._install_redo_shortcuts()

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

        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self.toggle_playback)
        self.spin_fps = QtWidgets.QDoubleSpinBox()
        self.spin_fps.setDecimals(1)
        self.spin_fps.setRange(1.0, 60.0)
        self.spin_fps.setSingleStep(1.0)
        self.spin_fps.setValue(self.play_fps)
        self.spin_fps.valueChanged.connect(self._on_fps_changed)

        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(self.btn_play)
        slider_row.addWidget(QtWidgets.QLabel("FPS"))
        slider_row.addWidget(self.spin_fps)
        slider_row.addWidget(self.slider, stretch=1)
        layout.addLayout(slider_row, 2, 0, 1, 1)

        # wheel filter
        self._wheel_filter = _WheelToSliderFilter(self.slider)
        for v in self.cine_views.values():
            v.installEventFilter(self._wheel_filter)
        self.pcmra_view.installEventFilter(self._wheel_filter)
        self.vel_view.installEventFilter(self._wheel_filter)
        self.slider.installEventFilter(self._wheel_filter)
        self.pcmra_view.getView().scene().installEventFilter(self)
        self.vel_view.getView().scene().installEventFilter(self)

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
        self.spin_line_angle = QtWidgets.QDoubleSpinBox()
        self.spin_line_angle.setDecimals(1)
        self.spin_line_angle.setRange(-90.0, 90.0)
        self.spin_line_angle.setSingleStep(1.0)
        self.spin_line_angle.setValue(0.0)
        line_ctrl_row.addWidget(self.btn_line_copy)
        line_ctrl_row.addWidget(self.btn_line_paste)
        line_ctrl_row.addWidget(self.btn_line_forward)
        line_ctrl_row.addWidget(QtWidgets.QLabel("Angle (deg)"))
        line_ctrl_row.addWidget(self.spin_line_angle)
        line_ctrl_row.addStretch(1)
        left_controls_layout.addLayout(line_ctrl_row)
        self.btn_line_copy.clicked.connect(self.copy_line_state)
        self.btn_line_paste.clicked.connect(self.paste_line_state)
        self.btn_line_forward.clicked.connect(self.copy_line_forward)
        self.spin_line_angle.valueChanged.connect(self.on_line_angle_changed)

        axis_row = QtWidgets.QHBoxLayout()
        axis_row.addWidget(QtWidgets.QLabel("Axis order"))
        axis_row.addWidget(self.axis_order_selector)
        axis_row.addWidget(self.chk_axis_flip_x)
        axis_row.addWidget(self.chk_axis_flip_y)
        axis_row.addWidget(self.chk_axis_flip_z)
        self.btn_cine_inference = QtWidgets.QPushButton("Inference")
        axis_row.addWidget(self.btn_cine_inference)
        self.btn_line_apply = QtWidgets.QPushButton("Apply")
        axis_row.addWidget(self.btn_line_apply)
        axis_row.addStretch(1)
        left_controls_layout.addLayout(axis_row)

        for btn in (
            self.btn_roi_copy,
            self.btn_roi_paste,
            self.btn_roi_forward,
            self.btn_line_copy,
            self.btn_line_paste,
            self.btn_line_forward,
            self.btn_cine_gif,
            self.btn_pcmra_gif,
            self.btn_vel_gif,
            self.btn_copy_regional,
            self.btn_cine_inference,
            self.btn_line_apply,
        ):
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.btn_load_mvpack.clicked.connect(self.on_load_mvpack)
        self.btn_cine_inference.clicked.connect(self.on_cine_inference)
        self.btn_line_apply.clicked.connect(self.on_line_apply)
        self._configure_cinema_inference_widget()

        self.pcmra_view.getView().scene().sigMouseClicked.connect(self.on_anchor_pick_pcmra)
        self._play_timer.timeout.connect(self._advance_playback)

        # try restore MVtrack.h5
        if restore_state:
            self.try_restore_state()

        self._emit_geometry_debug()
        shrink_callback = self._shrink_refine_options_width
        QtCore.QTimer.singleShot(0, shrink_callback)

        self._update_cine_roi_visibility()
        self.set_phase(0)
        self.apply_cine_levels()
        self.apply_pcmra_levels()
        self.apply_level_range()
        self._apply_levels_if_set()
        if self._restored_state:
            self.compute_all()
            self.update_plot_for_selection()

    def _configure_pcmra_refine_widget(self) -> None:
        reason = None
        try:
            settings = get_medsam2_settings()
        except Exception as exc:
            settings = {}
            reason = f"MedSAM2 settings failed to load: {exc}"

        if reason is None:
            missing = []
            python_path = settings.get("python")
            runner_path = settings.get("runner")
            ckpt_path = settings.get("checkpoint")
            if not python_path or not os.path.exists(str(python_path)):
                missing.append("python not found")
            if not runner_path or not os.path.exists(str(runner_path)):
                missing.append("runner not found")
            if not ckpt_path or not os.path.exists(str(ckpt_path)):
                missing.append("checkpoint not found")
            if missing:
                reason = "MedSAM2 unavailable: " + ", ".join(missing)

        if reason:
            for widget in [self.chk_negative_points, self.btn_refine_roi_phase, self.btn_refine_roi_all]:
                widget.setEnabled(False)
                widget.setToolTip(reason)
            self.pcmra_refine_widget.setToolTip(reason)

    def _configure_cinema_inference_widget(self) -> None:
        if not hasattr(self, "btn_cine_inference"):
            return
        reason = None
        try:
            settings = get_cinema_settings()
        except Exception as exc:
            settings = {}
            reason = f"CineMA settings failed to load: {exc}"

        if reason is None:
            missing = []
            python_path = settings.get("python")
            runner_path = settings.get("runner")
            if not python_path or not os.path.exists(str(python_path)):
                missing.append("python not found")
            if not runner_path or not os.path.exists(str(runner_path)):
                missing.append("runner not found")
            if missing:
                reason = "CineMA unavailable: " + ", ".join(missing)

        if reason:
            self.btn_cine_inference.setEnabled(False)
            self.btn_cine_inference.setToolTip(reason)

    def _shrink_refine_options_width(self) -> None:
        if not hasattr(self, "pcmra_refine_widget"):
            return
        self.pcmra_refine_widget.adjustSize()
        size_hint = self.pcmra_refine_widget.sizeHint()
        size_policy = self.pcmra_refine_widget.sizePolicy()
        self.pcmra_refine_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            size_policy.verticalPolicy(),
        )
        self.pcmra_refine_widget.setMaximumWidth(size_hint.width())
        self.pcmra_refine_widget.updateGeometry()

    # ============================
    # Restore state
    # ============================
    def _prompt_mvtrack(self, folder: str) -> Tuple[Optional[str], bool]:
        mvtrack_candidates = find_mvtrack_files(folder)
        tracking_path = None
        restore_state = False

        if mvtrack_candidates:
            mvtrack_candidates = sorted(
                mvtrack_candidates,
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )

            choices = ["(new tracking)"] + [os.path.basename(p) for p in mvtrack_candidates]
            default_index = 1

            choice, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Select MVtrack",
                "Select an existing MVtrack file to load:",
                choices,
                default_index,
                False,
            )

            if ok and choice and choice != "(new tracking)":
                for p in mvtrack_candidates:
                    if os.path.basename(p) == choice:
                        tracking_path = p
                        restore_state = True
                        break

        return tracking_path, restore_state

    def on_load_mvpack(self):
        mvpack_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select mvpack.h5",
            self.work_folder,
            "HDF5 Files (*.h5)",
        )
        if not mvpack_path:
            return
        try:
            pack = load_mvpack_h5(mvpack_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "MV tracker", f"Failed to load mvpack:\n{exc}")
            return

        folder = os.path.dirname(mvpack_path)
        tracking_path, restore_state = self._prompt_mvtrack(folder)
        new_window = ValveTracker(
            pack,
            work_folder=folder,
            tracking_path=tracking_path,
            restore_state=restore_state,
        )
        new_window.resize(self.width(), self.height())
        new_window.show()
        self._child_windows.append(new_window)
        self.close()

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
        self.line_angle = st.get("line_angle", self.line_angle)
        self.roi_state = st.get("roi_state", self.roi_state)
        self.roi_locked = st.get("roi_locked", self.roi_locked)
        try:
            segment_payload = json.loads(st.get("segment_payload_json", "{}"))
            if isinstance(segment_payload, dict):
                self.metrics_seg4 = segment_payload.get("seg4", self.metrics_seg4)
                self.metrics_seg6 = segment_payload.get("seg6", self.metrics_seg6)
        except Exception:
            pass
        try:
            self._segment_ref_angle = st.get("segment_ref_angle", self._segment_ref_angle)
            self._segment_anchor_xy = st.get("segment_anchor_xy", self._segment_anchor_xy)
            self._segment_count = st.get("segment_count_list", self._segment_count)
        except Exception:
            pass
        segment_count = int(st.get("segment_count", 6))
        plot_segments = bool(int(st.get("plot_segments", 0)))
        apply_segments = bool(int(st.get("apply_segments", 0)))
        show_segment_labels = bool(int(st.get("show_segment_labels", 0)))
        flip_flow = bool(int(st.get("flip_flow", 0)))
        axis_order = str(st.get("axis_order", "XYZ"))
        axis_flips_json = st.get("axis_flips_json", "[false, false, false]")
        try:
            axis_flips = json.loads(axis_flips_json)
        except Exception:
            axis_flips = [False, False, False]
        self.segment_selector.setCurrentText("6 segments" if segment_count == 6 else "4 segments")
        self.chk_plot_segments.setChecked(plot_segments)
        self.chk_apply_segments.setChecked(apply_segments)
        self.chk_segment_labels.setChecked(show_segment_labels)
        self.chk_flip_flow.setChecked(flip_flow)
        if self.metrics_seg4 or self.metrics_seg6:
            self.chk_apply_segments.setChecked(True)
        self._show_segments = bool(self.chk_apply_segments.isChecked())
        self._update_segment_context_menu()

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
        except Exception:
            display_levels = {}
            vel_auto_once = {}
            cine_levels = [None, None]
            pcmra_levels = [None, None]
        cine_auto_once = bool(st.get("cine_auto_once", 1))
        pcmra_auto_once = bool(st.get("pcmra_auto_once", 1))

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

        self._sync_level_controls_from_state()

        axis_choices = [self.axis_order_selector.itemText(i) for i in range(self.axis_order_selector.count())]
        if axis_order not in axis_choices:
            axis_order = "XYZ"
        self.axis_order = axis_order
        self.axis_flips = tuple(bool(val) for val in list(axis_flips)[:3] + [False, False, False])[:3]
        self.axis_order_selector.blockSignals(True)
        self.chk_axis_flip_x.blockSignals(True)
        self.chk_axis_flip_y.blockSignals(True)
        self.chk_axis_flip_z.blockSignals(True)
        try:
            self.axis_order_selector.setCurrentText(self.axis_order)
            self.chk_axis_flip_x.setChecked(self.axis_flips[0])
            self.chk_axis_flip_y.setChecked(self.axis_flips[1])
            self.chk_axis_flip_z.setChecked(self.axis_flips[2])
        finally:
            self.axis_order_selector.blockSignals(False)
            self.chk_axis_flip_x.blockSignals(False)
            self.chk_axis_flip_y.blockSignals(False)
            self.chk_axis_flip_z.blockSignals(False)

        self.memo.appendPlainText(f"Restored tracking state from: {st_path}")

        self.compute_all()
        self.update_plot_for_selection()

    # ============================
    # Cine helpers
    # ============================
    def _get_cine_display_map(self, cine_key: str, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        cache = self._cine_display_maps.get(cine_key)
        if cache is not None and cache[0] == shape:
            return cache[1], cache[2]
        cine_geom = self._get_cine_geom_raw(cine_key)
        rowq, colq = cine_display_mapping(cine_geom, shape)
        self._cine_display_maps[cine_key] = (shape, rowq, colq)
        return rowq, colq

    def _get_cine_frame(self, cine_key: str, t: int) -> np.ndarray:
        img_raw = self._get_cine_frame_raw(cine_key, t)
        rowq, colq = self._get_cine_display_map(cine_key, img_raw.shape)
        coords = np.vstack([rowq.ravel(), colq.ravel()])
        img_disp = map_coordinates(
            img_raw.astype(np.float32),
            coords,
            order=1,
            mode="constant",
            cval=0.0,
        ).reshape(img_raw.shape)
        return img_disp

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

    def _get_cine_geom_raw(self, cine_key: str) -> CineGeom:
        return self.pack.cine_planes[cine_key]["geom"]

    def _cine_slice_thickness_mm(self, cine_geom: CineGeom) -> Optional[float]:
        edges = cine_geom.edges
        if edges is None:
            return None
        edges = np.asarray(edges, dtype=np.float64)
        if edges.shape[0] < 3 or edges.shape[1] < 3:
            return None
        thickness = float(np.linalg.norm(edges[:3, 2]))
        if not np.isfinite(thickness) or thickness <= 0.0:
            return None
        return thickness

    def on_cine_inference(self):
        cine_key = self.active_cine_key
        if cine_key not in self.pack.cine_planes:
            self.memo.appendPlainText("[mvtracking] No cine selected for inference.")
            return
        view = "lax_2c" if cine_key == "2ch" else "lax_4c"
        self.memo.appendPlainText(f"[mvtracking] Inference cine key: {cine_key}")
        self.memo.appendPlainText(f"[mvtracking] Inference view: {view}")
        try:
            frames = [self._get_cine_frame(cine_key, t) for t in range(self.Nt)]
            cine_stack = np.stack(frames, axis=0).astype(np.float32)
            start = time.perf_counter()
            output, stdout, stderr = run_cinema_subprocess(cine_stack, view)
            elapsed = time.perf_counter() - start
        except Exception as exc:
            self.memo.appendPlainText(f"[mvtracking] Inference failed: {exc}")
            return

        try:
            T, H, W = cine_stack.shape
            self.memo.appendPlainText(f"[mvtracking] Input frame shape: ({T}, {H}, {W})")
            self.memo.appendPlainText(f"[mvtracking] Inference time: {elapsed:.3f}s")
            if stdout:
                self.memo.appendPlainText("[mvtracking] CineMA stdout captured.")
            if stderr:
                self.memo.appendPlainText("[mvtracking] CineMA stderr captured.")

            coords = np.asarray(output.get("coords", []), dtype=np.float32)
            if coords.ndim != 2:
                raise RuntimeError("CineMA output coords must be 2D.")
            if coords.shape == (T, 6):
                coords = coords.T
            if coords.shape != (6, T):
                raise RuntimeError(f"CineMA output coords shape invalid: {coords.shape}")

            mv = output.get("mv")
            mv_arr = None if mv is None else np.asarray(mv, dtype=np.float32)
            if mv_arr is None or mv_arr.size == 0:
                mv_arr = coords[2:6, :]
            elif mv_arr.ndim != 2:
                raise RuntimeError("CineMA output mv must be 2D.")
            elif mv_arr.shape == (T, 4):
                mv_arr = mv_arr.T
            if mv_arr.shape != (4, T):
                raise RuntimeError(f"CineMA output mv shape invalid: {mv_arr.shape}")

            x1 = mv_arr[0]
            y1 = mv_arr[1]
            x2 = mv_arr[2]
            y2 = mv_arr[3]
            mv_pts_t = np.stack(
                [np.stack([x1, y1], axis=-1), np.stack([x2, y2], axis=-1)],
                axis=1,
            )
        except Exception as exc:
            self.memo.appendPlainText(f"[mvtracking] Inference output invalid: {exc}")
            return

        for t in range(T):
            img_raw = self._get_cine_frame_raw(cine_key, t)
            H_raw, W_raw = img_raw.shape
            line_abs = self._canonicalize_line_abs(mv_pts_t[t])
            self.line_norm[t] = self._abs_to_norm_line(line_abs, H_raw, W_raw)

        cur_t = int(self.slider.value()) - 1
        if 0 <= cur_t < self.Nt:
            self.set_phase(cur_t, force=True)

        x_min = float(np.min(mv_pts_t[:, :, 0]))
        x_max = float(np.max(mv_pts_t[:, :, 0]))
        y_min = float(np.min(mv_pts_t[:, :, 1]))
        y_max = float(np.max(mv_pts_t[:, :, 1]))
        self.memo.appendPlainText(f"[mvtracking] MV coord range: x=[{x_min:.1f}, {x_max:.1f}] y=[{y_min:.1f}, {y_max:.1f}]")

        lengths = np.linalg.norm(mv_pts_t[:, 0, :] - mv_pts_t[:, 1, :], axis=1)
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))
        cv = std_len / max(mean_len, 1e-6)
        quality = "OK" if cv < 0.3 else "WARN"
        self.memo.appendPlainText(
            f"[mvtracking] MV length check: mean={mean_len:.2f} std={std_len:.2f} cv={cv:.2f} => {quality}"
        )

    def copy_line_state(self):
        t = int(self.slider.value()) - 1
        self._sync_current_phase_state()
        st = self.line_norm[t]
        if st is None:
            return
        self._line_clipboard = np.array(st, dtype=np.float64).copy()
        self._line_angle_clipboard = float(self.line_angle[t])
        self.memo.appendPlainText("Copied cine line.")

    def paste_line_state(self):
        if self._line_clipboard is None:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            self.memo.appendPlainText("ROI is locked for this phase.")
            return
        self.line_norm[t] = np.array(self._line_clipboard, dtype=np.float64).copy()
        if self._line_angle_clipboard is not None:
            self.line_angle[t] = float(self._line_angle_clipboard)
            self.spin_line_angle.blockSignals(True)
            self.spin_line_angle.setValue(float(self._line_angle_clipboard))
            self.spin_line_angle.blockSignals(False)
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
                self.line_angle[tt] = float(self.line_angle[t])
                self._segment_ref_angle[tt] = self._segment_ref_angle[t]
                self._segment_anchor_xy[tt] = (
                    None if self._segment_anchor_xy[t] is None else np.array(self._segment_anchor_xy[t]).copy()
                )
                self._segment_count[tt] = self._segment_count[t]
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

    def _default_line_norm(self) -> np.ndarray:
        return np.array([[0.45, 0.55], [0.60, 0.45]], dtype=np.float64)

    def on_cine_line_changed_live(self, cine_key: str):
        if self._syncing_cine_line or cine_key != self.active_cine_key:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            return
        self._begin_history_capture("line", t)

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
        self._commit_history_capture("line", t)
        self._update_line_marker_overlay(t)

    def _update_cine_roi_visibility(self):
        for k, roi in self.cine_line_rois.items():
            roi.setVisible(k == self.active_cine_key)

    def _update_cine_view_visibility(self, cine_key: str):
        view = self.cine_views.get(cine_key)
        chk = self.cine_view_checks.get(cine_key)
        if view is None or chk is None:
            return
        visible = bool(chk.isChecked())
        view.setVisible(visible)
        if visible:
            img_item = view.getImageItem()
            if img_item is not None and img_item.image is not None:
                h, w = img_item.image.shape[:2]
                view.getView().setRange(xRange=(0, w), yRange=(0, h), padding=0.0)
            self._view_ranges[f"cine:{cine_key}"] = None
            if sum(1 for c in self.cine_view_checks.values() if c.isChecked()) == 1:
                try:
                    view.getView().autoRange()
                except Exception:
                    pass

    def _emit_geometry_debug(self):
        geom = self.pack.geom
        axis_map = geom.axis_map if geom.axis_map is not None else {}
        print(
            "[mvtracking] geom axis_map="
            f"{axis_map} orgn4={np.array2string(geom.orgn4, precision=4, separator=',')} "
            f"A0={np.array2string(geom.A[:, 0], precision=4, separator=',')}"
        )
        if geom.slice_order is not None:
            order = geom.slice_order
            print(f"[mvtracking] slice_order first/last={int(order[0])}/{int(order[-1])}")
        if geom.slice_positions is not None:
            sp = geom.slice_positions
            print(f"[mvtracking] slice_positions first/last={float(sp[0]):.4f}/{float(sp[-1]):.4f}")

        if self.active_cine_key in self.pack.cine_planes:
            cine_geom = self._get_cine_geom_raw(self.active_cine_key)
            col_vec = cine_geom.iop[:3]
            row_vec = cine_geom.iop[3:6]
            normal_vec = np.cross(col_vec, row_vec)
            nn = np.linalg.norm(normal_vec)
            normal_vec = normal_vec / (nn if nn > 0 else 1e-12)
            print(
                "[mvtracking] cine axis_map="
                f"{self.pack.cine_planes[self.active_cine_key]['geom'].axis_map} "
                f"normal={np.array2string(normal_vec, precision=4, separator=',')}"
            )

    # ============================
    # Phase update
    # ============================
    def on_phase_changed(self, v: int):
        self._remember_current_levels(self._current_display_mode)
        self.set_phase(v - 1)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.toggle_playback(not self.btn_play.isChecked())
            return
        if isinstance(self.focusWidget(), (QtWidgets.QAbstractSpinBox, QtWidgets.QLineEdit)):
            super().keyPressEvent(event)
            return
        if event.key() == QtCore.Qt.Key.Key_N:
            self._set_negative_point_mode(not self._negative_point_mode)
            return
        if event.key() == QtCore.Qt.Key.Key_2:
            self.toggle_brush_mode()
            return
        if event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Down):
            self.slider.setValue(max(self.slider.minimum(), self.slider.value() - 1))
            return
        if event.key() in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Up):
            self.slider.setValue(min(self.slider.maximum(), self.slider.value() + 1))
            return
        super().keyPressEvent(event)

    def set_phase(self, t: int, force: bool = False):
        t = int(np.clip(t, 0, self.Nt - 1))
        if self._cur_phase == t and not force:
            return
        self._cur_phase = t
        self.lbl_phase.setText(f"Phase: {t + 1}")
        if 0 <= t < len(self.line_angle):
            self.spin_line_angle.blockSignals(True)
            self.spin_line_angle.setValue(float(self.line_angle[t]))
            self.spin_line_angle.blockSignals(False)
        if 0 <= t < len(self._segment_count):
            self.segment_selector.blockSignals(True)
            self.segment_selector.setCurrentText(
                "6 segments" if self._segment_count[t] == 6 else "4 segments"
            )
            self.segment_selector.blockSignals(False)
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

        # Auto-level captures update the stored ranges; sync controls afterwards so
        # the UI reflects the automatic windowing instead of the 0-1 placeholders.
        self._sync_level_controls_from_state()
        self._apply_levels_if_set()

        if self.roi_state[t] is None:
            self.roi_state[t] = self.default_poly_roi_state_from_image(Ivelmag)

        self.ensure_poly_rois()
        self.apply_roi_state_both(self.roi_state[t])
        self.update_spline_overlay(t)
        self._update_negative_point_overlay(t)
        self.update_metric_labels(t)
        self._update_roi_lock_ui()
        self.set_poly_editable(self.edit_mode and not self._is_roi_locked(t))
        self._update_line_editable(t)
        self._update_line_marker_overlay(t)
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
        img_raw = self._get_cine_frame_raw(self.active_cine_key, t)
        angle_deg = float(self.line_angle[t]) if 0 <= t < len(self.line_angle) else float(self.spin_line_angle.value())

        extra_scalars: Dict[str, np.ndarray] = {}
        if self.pack.ke is not None and self.pack.ke.ndim == 4:
            extra_scalars["ke"] = self.pack.ke[:, :, :, t].astype(np.float32)
        if self.pack.vortmag is not None and self.pack.vortmag.ndim == 4:
            extra_scalars["vortmag"] = self.pack.vortmag[:, :, :, t].astype(np.float32)

        if self.axis_order or self.axis_flips:
            pcmra3d = apply_axis_transform(pcmra3d, self.axis_order, self.axis_flips)
            vel5d = apply_axis_transform(vel5d, self.axis_order, self.axis_flips)
            vel5d = transform_vector_components(vel5d, self.axis_order, self.axis_flips)
            for key, vol in list(extra_scalars.items()):
                extra_scalars[key] = apply_axis_transform(vol, self.axis_order, self.axis_flips)

        Ipcm, Ivelmag, Vn, spmm, extras = reslice_plane_fixedN(
            pcmra3d=pcmra3d,
            vel5d=vel5d,
            t=t,
            vol_geom=self.pack.geom,
            cine_geom=cine_geom,
            line_xy=line_xy,
            cine_shape=img_raw.shape,
            angle_offset_deg=angle_deg,
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

    def _on_axis_transform_changed(self, _value: Optional[int] = None):
        self.axis_order = self.axis_order_selector.currentText()
        self.axis_flips = (
            self.chk_axis_flip_x.isChecked(),
            self.chk_axis_flip_y.isChecked(),
            self.chk_axis_flip_z.isChecked(),
        )
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return
        self._cur_phase = None
        self.set_phase(t)
        self.compute_current(update_only=True)

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
        cine_geom = self._get_cine_geom_raw(self.active_cine_key)
        cine_img_raw = self._get_cine_frame_raw(self.active_cine_key, t)
        angle_deg = float(self.line_angle[t]) if 0 <= t < len(self.line_angle) else float(self.spin_line_angle.value())
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
        contour_pts = self._roi_contour_points((self.Npix, self.Npix))
        if contour_pts is None or contour_pts.size == 0:
            self.memo.appendPlainText("STL save failed: no PCMRA contour available.")
            return
        contour_pts = contour_pts.copy()
        contour_pts[:, 1] = (self.Npix - 1) - contour_pts[:, 1]
        contour_xyz = self._pcmra_contour_patient_xyz(
            contour_pts=contour_pts,
            line_xy=line_xy,
            cine_geom=cine_geom,
            cine_shape=cine_img_raw.shape,
            angle_deg=angle_deg,
        )
        if contour_xyz is None or contour_xyz.size == 0:
            self.memo.appendPlainText("STL save failed: unable to map contour to patient space.")
            return
        thickness_mm = self._cine_slice_thickness_mm(cine_geom)
        if thickness_mm is None:
            write_stl_from_patient_contour(
                out_path=out_path,
                contour_pts_xyz=contour_xyz,
                output_space="LPS",
            )
        else:
            write_stl_from_patient_contour_extruded(
                out_path=out_path,
                contour_pts_xyz=contour_xyz,
                thickness_mm=thickness_mm,
                output_space="LPS",
            )
        if not os.path.exists(out_path):
            self.memo.appendPlainText(f"STL save failed: {out_path}")
            return
        self.memo.appendPlainText(
            f"STL saved: {out_path} (patient/PCMRA contour in LPS; set output_space='RAS' for RAS)."
        )

    def _pcmra_contour_patient_xyz(
        self,
        contour_pts: np.ndarray,
        line_xy: np.ndarray,
        cine_geom: CineGeom,
        cine_shape: Tuple[int, int],
        angle_deg: float,
    ) -> Optional[np.ndarray]:
        if contour_pts is None or contour_pts.size == 0:
            return None
        if self.Npix <= 1:
            return None
        c, u, v, _ = make_plane_from_cine_line(
            line_xy,
            cine_geom,
            cine_shape=cine_shape,
            angle_offset_deg=angle_deg,
        )
        fov_half = auto_fov_from_line(line_xy, cine_geom)
        scale = (2.0 * fov_half) / float(self.Npix - 1)
        vv = (-fov_half + contour_pts[:, 0] * scale).reshape(-1, 1)
        uu = (-fov_half + (self.Npix - 1 - contour_pts[:, 1]) * scale).reshape(-1, 1) # SP edited for converting STL
        return c[None, :] + uu * u[None, :] + vv * v[None, :]

    def _normalize_image(self, img: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
        data = img.astype(np.float64)
        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            return np.zeros_like(data, dtype=np.uint8)
        data = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
        return (data * 255.0).astype(np.uint8)

    def _draw_line(self, img: np.ndarray, p0: np.ndarray, p1: np.ndarray, color: Tuple[int, int, int]) -> None:
        h, w = img.shape[:2]
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if 0 <= x0 < w and 0 <= y0 < h:
                img[y0, x0, :] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _draw_polyline(self, img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int]) -> None:
        if pts is None or pts.shape[0] < 2:
            return
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            self._draw_line(img, p0, p1, color)

    def _grab_view_frame(self, widget: QtWidgets.QWidget) -> np.ndarray:
        pixmap = widget.grab()
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        w = image.width()
        h = image.height()
        bytes_per_line = image.bytesPerLine()
        buf = image.bits().tobytes()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, bytes_per_line // 4, 4))
        return arr[:, :w, :3].copy()

    def _render_widget_for_capture(self, widget: QtWidgets.QWidget) -> None:
        widget.repaint()
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)

    def _export_view_gif(self, widget: QtWidgets.QWidget, title: str, default_name: str) -> None:
        if imageio is None:
            QtWidgets.QMessageBox.warning(self, "MV tracker", "imageio is not installed.")
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            os.path.join(self.work_folder, default_name),
            "GIF Files (*.gif)",
        )
        if not out_path:
            return
        frames = []
        current_phase = int(self.slider.value()) - 1
        for tt in range(self.Nt):
            self.set_phase(tt)
            QtWidgets.QApplication.processEvents()
            self._render_widget_for_capture(widget)
            frames.append(self._grab_view_frame(widget))
        self.set_phase(current_phase)
        duration = 1.0 / max(float(self.spin_fps.value()), 1.0)
        imageio.mimsave(out_path, frames, duration=duration, loop=0)
        self.memo.appendPlainText(f"GIF saved: {out_path}")

    def export_cine_gif(self):
        if self.active_cine_key not in self.pack.cine_planes:
            return
        view = self.cine_views.get(self.active_cine_key)
        if view is None:
            return
        self._export_view_gif(view, "Save Cine GIF", f"cine_{self.active_cine_key}.gif")

    def export_pcmra_gif(self):
        self._export_view_gif(self.pcmra_view, "Save PCMRA GIF", "pcmra_view.gif")

    def export_vel_gif(self):
        self._export_view_gif(self.vel_view, "Save Colormap GIF", "colormap_view.gif")

    # ============================
    # ROI state
    # ============================
    def default_poly_roi_state(self, shape_hw: Tuple[int, int]) -> dict:
        H, W = shape_hw
        cx, cy = W * 0.5, H * 0.5
        r = min(H, W) * 0.12
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
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
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
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

        if not hasattr(self, "segment_curve_pcm") or self.segment_curve_pcm is None:
            self.segment_curve_pcm = pg.PlotCurveItem(pen=pg.mkPen("m", width=1))
            self.pcmra_view.getView().addItem(self.segment_curve_pcm)

        if not hasattr(self, "segment_curve_vel") or self.segment_curve_vel is None:
            self.segment_curve_vel = pg.PlotCurveItem(pen=pg.mkPen("m", width=1))
            self.vel_view.getView().addItem(self.segment_curve_vel)

        if not hasattr(self, "segment_anchor_marker") or self.segment_anchor_marker is None:
            self.segment_anchor_marker = pg.ScatterPlotItem(
                size=8,
                pen=pg.mkPen("m"),
                brush=pg.mkBrush("m"),
            )
            self.pcmra_view.getView().addItem(self.segment_anchor_marker)

        if not hasattr(self, "_negative_point_marker") or self._negative_point_marker is None:
            self._negative_point_marker = pg.ScatterPlotItem(
                size=10,
                pen=pg.mkPen((255, 80, 80)),
                brush=pg.mkBrush(255, 80, 80, 120),
                symbol="x",
            )
            self.pcmra_view.getView().addItem(self._negative_point_marker)

        if not hasattr(self, "line_marker_pcm") or self.line_marker_pcm is None:
            self.line_marker_pcm = pg.ScatterPlotItem(
                size=12,
                pen=pg.mkPen((255, 220, 0), width=2),
                brush=pg.mkBrush(0, 0, 0, 0),
                symbol="o",
            )
            self.pcmra_view.getView().addItem(self.line_marker_pcm)

        if not hasattr(self, "line_marker_vel") or self.line_marker_vel is None:
            self.line_marker_vel = pg.ScatterPlotItem(
                size=12,
                pen=pg.mkPen((255, 220, 0), width=2),
                brush=pg.mkBrush(0, 0, 0, 0),
                symbol="o",
            )
            self.vel_view.getView().addItem(self.line_marker_vel)

        if not self._segment_label_items_pcm:
            for _ in range(6):
                item = pg.TextItem("", color="m", anchor=(0.5, 0.5))
                item.setVisible(False)
                self.pcmra_view.getView().addItem(item)
                self._segment_label_items_pcm.append(item)
        if not self._segment_label_items_vel:
            for _ in range(6):
                item = pg.TextItem("", color="w", anchor=(0.5, 0.5))
                item.setVisible(False)
                self.vel_view.getView().addItem(item)
                self._segment_label_items_vel.append(item)

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
        self._update_segment_overlay(t)

    def on_line_apply(self) -> None:
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return
        self._line_marker_enabled = True
        self.ensure_poly_rois()
        self._update_line_marker_overlay(t)

    def _line_marker_positions(self, t: int) -> Optional[np.ndarray]:
        if t < 0 or t >= self.Nt:
            return None
        if self.line_norm[t] is None:
            return None
        line_xy = self._get_active_line_abs_raw(t)
        if line_xy is None or line_xy.shape != (2, 2):
            return None
        cine_geom = self._get_cine_geom_raw(self.active_cine_key)
        img_raw = self._get_cine_frame_raw(self.active_cine_key, t)
        angle_deg = float(self.line_angle[t]) if 0 <= t < len(self.line_angle) else float(self.spin_line_angle.value())
        try:
            pts_xyz = cine_line_to_patient_xyz(line_xy, cine_geom, cine_shape=img_raw.shape)
            center, u, v, _ = make_plane_from_cine_line(
                line_xy,
                cine_geom,
                cine_shape=img_raw.shape,
                angle_offset_deg=angle_deg,
            )
        except Exception:
            return None
        fov_half = auto_fov_from_line(line_xy, cine_geom)
        if not np.isfinite(fov_half) or fov_half <= 0.0:
            return None
        rel = pts_xyz - center.reshape(1, 3)
        x_plane = rel @ u
        y_plane = rel @ v
        denom = 2.0 * fov_half
        row = (y_plane + fov_half) / denom * (self.Npix - 1)
        col = (x_plane + fov_half) / denom * (self.Npix - 1)
        x_disp = row
        y_disp = col
        return np.column_stack([x_disp, y_disp])

    def _update_line_marker_overlay(self, t: int) -> None:
        if self.line_marker_pcm is None or self.line_marker_vel is None:
            return
        if not self._line_marker_enabled:
            self.line_marker_pcm.setData([], [])
            self.line_marker_vel.setData([], [])
            return
        pts = self._line_marker_positions(t)
        if pts is None or len(pts) == 0:
            self.line_marker_pcm.setData([], [])
            self.line_marker_vel.setData([], [])
            return
        xs = pts[:, 0].astype(np.float64)
        ys = pts[:, 1].astype(np.float64)
        self.line_marker_pcm.setData(xs, ys)
        self.line_marker_vel.setData(xs, ys)

    def on_poly_changed_pcm_live(self):
        if self._syncing_poly:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            return
        self._begin_history_capture("roi", t)
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
        self._commit_history_capture("roi", t)

    def on_poly_changed_vel_live(self):
        if self._syncing_poly:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            return
        self._begin_history_capture("roi", t)
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
        self._commit_history_capture("roi", t)

    # ============================
    # Metrics
    # ============================
    def toggle_edit(self):
        self.edit_mode = not self.edit_mode
        self.btn_edit.setText("Edit ROI: ON" if self.edit_mode else "Edit ROI: OFF")
        t = int(self.slider.value()) - 1
        self.set_poly_editable(self.edit_mode and not self._is_roi_locked(t))

    def _begin_history_capture(self, kind: str, phase: int) -> None:
        if self._restoring_history or phase < 0 or phase >= self.Nt:
            return
        if self._history_active is not None:
            if self._history_active.get("kind") == kind and self._history_active.get("phase") == phase:
                return
            return
        state = self._snapshot_history_state(kind, phase)
        self._history_active = {"kind": kind, "phase": phase, "state": state}

    def _commit_history_capture(self, kind: str, phase: int) -> None:
        if self._restoring_history:
            return
        if self._history_active is None:
            return
        if self._history_active.get("kind") != kind or self._history_active.get("phase") != phase:
            return
        previous = self._history_active.get("state")
        self._history_active = None
        current = self._snapshot_history_state(kind, phase)
        if self._history_states_equal(kind, previous, current):
            return
        self._undo_stack.append({"kind": kind, "phase": phase, "state": previous})
        self._redo_stack.clear()

    def _snapshot_history_state(self, kind: str, phase: int):
        if kind == "line":
            state = self.line_norm[phase]
            return None if state is None else np.array(state, dtype=np.float64).copy()
        if kind == "roi":
            state = self.roi_state[phase]
            return None if state is None else json.loads(json.dumps(state))
        return None

    def _history_states_equal(self, kind: str, a, b) -> bool:
        if kind == "line":
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return np.allclose(np.array(a), np.array(b))
        if kind == "roi":
            return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        return False

    def _apply_history_state(self, entry: dict) -> None:
        kind = entry.get("kind")
        phase = entry.get("phase")
        if kind not in ("line", "roi") or phase is None:
            return
        self._restoring_history = True
        try:
            if kind == "line":
                state = entry.get("state")
                self.line_norm[phase] = None if state is None else np.array(state, dtype=np.float64).copy()
                if phase == int(self.slider.value()) - 1:
                    self._cur_phase = None
                    self.set_phase(phase)
            elif kind == "roi":
                state = entry.get("state")
                self.roi_state[phase] = None if state is None else json.loads(json.dumps(state))
                if phase == int(self.slider.value()) - 1 and state is not None:
                    self.apply_roi_state_both(state)
                    self.update_spline_overlay(phase)
                    self.compute_current(update_only=True)
        finally:
            self._restoring_history = False

    def undo_last_action(self):
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        current = self._snapshot_history_state(entry["kind"], entry["phase"])
        self._redo_stack.append({**entry, "state": current})
        self._apply_history_state(entry)

    def redo_last_action(self):
        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        current = self._snapshot_history_state(entry["kind"], entry["phase"])
        self._undo_stack.append({**entry, "state": current})
        self._apply_history_state(entry)

    def _install_redo_shortcuts(self) -> None:
        widgets = list(self.cine_views.values()) + [self.pcmra_view, self.vel_view]
        for widget in widgets:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), widget)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self.redo_last_action)
            self._redo_shortcuts.append(shortcut)

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
        mask, center = self._roi_mask_and_center(Ivelmag.shape)
        ref_angle = self._segment_ref_angle[t] if 0 <= t < self.Nt else None
        if ref_angle is not None and center is not None and np.any(mask) and self._show_segments:
            for metric in (
                "Flow rate (mL/s)",
                "Peak velocity (m/s)",
                "Mean velocity (m/s)",
                "Kinetic energy (uJ)",
                "Peak vorticity (1/s)",
                "Mean vorticity (1/s)",
            ):
                self.metrics_seg4.setdefault(metric, [None] * self.Nt)[t] = self._segment_values_for_metric(
                    metric,
                    Ivelmag,
                    Vn,
                    Ike,
                    Ivort,
                    spmm,
                    mask,
                    center,
                    ref_angle,
                    4,
                )
                self.metrics_seg6.setdefault(metric, [None] * self.Nt)[t] = self._segment_values_for_metric(
                    metric,
                    Ivelmag,
                    Vn,
                    Ike,
                    Ivort,
                    spmm,
                    mask,
                    center,
                    ref_angle,
                    6,
                )
        else:
            for metric in (
                "Flow rate (mL/s)",
                "Peak velocity (m/s)",
                "Mean velocity (m/s)",
                "Kinetic energy (uJ)",
                "Peak vorticity (1/s)",
                "Mean vorticity (1/s)",
            ):
                self.metrics_seg4.setdefault(metric, [None] * self.Nt)[t] = None
                self.metrics_seg6.setdefault(metric, [None] * self.Nt)[t] = None

        self.metrics_Q[t] = Q
        self.metrics_Vpk[t] = Vpk
        self.metrics_Vmn[t] = Vmn
        self.metrics_KE[t] = KE
        self.metrics_VortPk[t] = VortPk
        self.metrics_VortMn[t] = VortMn

        self.update_metric_labels(t)
        self._update_segment_overlay(t)

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
                for metric in (
                    "Flow rate (mL/s)",
                    "Peak velocity (m/s)",
                    "Mean velocity (m/s)",
                    "Kinetic energy (uJ)",
                    "Peak vorticity (1/s)",
                    "Mean vorticity (1/s)",
                ):
                    self.metrics_seg4.setdefault(metric, [None] * self.Nt)[t] = None
                    self.metrics_seg6.setdefault(metric, [None] * self.Nt)[t] = None
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

            ref_angle = self._segment_ref_angle[t] if 0 <= t < self.Nt else None
            if ref_angle is not None and self._show_segments:
                center = abs_pts.mean(axis=0)
                for metric in (
                    "Flow rate (mL/s)",
                    "Peak velocity (m/s)",
                    "Mean velocity (m/s)",
                    "Kinetic energy (uJ)",
                    "Peak vorticity (1/s)",
                    "Mean vorticity (1/s)",
                ):
                    self.metrics_seg4.setdefault(metric, [None] * self.Nt)[t] = self._segment_values_for_metric(
                        metric,
                        Ivelmag,
                        Vn,
                        Ike,
                        Ivort,
                        spmm,
                        mask,
                        center,
                        ref_angle,
                        4,
                    )
                    self.metrics_seg6.setdefault(metric, [None] * self.Nt)[t] = self._segment_values_for_metric(
                        metric,
                        Ivelmag,
                        Vn,
                        Ike,
                        Ivort,
                        spmm,
                        mask,
                        center,
                        ref_angle,
                        6,
                    )
            else:
                for metric in (
                    "Flow rate (mL/s)",
                    "Peak velocity (m/s)",
                    "Mean velocity (m/s)",
                    "Kinetic energy (uJ)",
                    "Peak vorticity (1/s)",
                    "Mean vorticity (1/s)",
                ):
                    self.metrics_seg4.setdefault(metric, [None] * self.Nt)[t] = None
                    self.metrics_seg6.setdefault(metric, [None] * self.Nt)[t] = None

            ref_angle = self._segment_ref_angle[t] if 0 <= t < self.Nt else None
            if ref_angle is not None and self._show_segments:
                center = abs_pts.mean(axis=0)
                for metric in (
                    "Flow rate (mL/s)",
                    "Peak velocity (m/s)",
                    "Mean velocity (m/s)",
                    "Kinetic energy (uJ)",
                    "Peak vorticity (1/s)",
                    "Mean vorticity (1/s)",
                ):
                    self.metrics_seg4.setdefault(metric, [None] * self.Nt)[t] = self._segment_values_for_metric(
                        metric,
                        Ivelmag,
                        Vn,
                        Ike,
                        Ivort,
                        spmm,
                        mask,
                        center,
                        ref_angle,
                        4,
                    )
                    self.metrics_seg6.setdefault(metric, [None] * self.Nt)[t] = self._segment_values_for_metric(
                        metric,
                        Ivelmag,
                        Vn,
                        Ike,
                        Ivort,
                        spmm,
                        mask,
                        center,
                        ref_angle,
                        6,
                    )
            else:
                for metric in (
                    "Flow rate (mL/s)",
                    "Peak velocity (m/s)",
                    "Mean velocity (m/s)",
                    "Kinetic energy (uJ)",
                    "Peak vorticity (1/s)",
                    "Mean vorticity (1/s)",
                ):
                    self.metrics_seg4.setdefault(metric, [None] * self.Nt)[t] = None
                    self.metrics_seg6.setdefault(metric, [None] * self.Nt)[t] = None

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

    def _roi_mask_and_center(self, shape_hw: Tuple[int, int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return np.zeros(shape_hw, dtype=bool), None
        st = self.roi_state[t]
        if st is None:
            st = self.default_poly_roi_state(shape_hw)
            self.roi_state[t] = st
        abs_pts = self._abs_pts_from_state_safe(st, shape_hw)
        if abs_pts.ndim != 2 or abs_pts.shape[0] < 3:
            return np.zeros(shape_hw, dtype=bool), None
        abs_pts = closed_spline_xy(abs_pts, n_out=400)
        H, W = shape_hw
        abs_pts[:, 0] = np.clip(abs_pts[:, 0], 0, W - 1)
        abs_pts[:, 1] = np.clip(abs_pts[:, 1], 0, H - 1)
        mask = polygon_mask(shape_hw, abs_pts)
        center = abs_pts.mean(axis=0) if abs_pts.size else None
        return mask, center

    def _roi_contour_points(self, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return None
        st = self.roi_state[t]
        if st is None:
            st = self.default_poly_roi_state(shape_hw)
            self.roi_state[t] = st
        abs_pts = self._abs_pts_from_state_safe(st, shape_hw)
        if abs_pts.ndim != 2 or abs_pts.shape[0] < 3:
            return None
        abs_pts = closed_spline_xy(abs_pts, n_out=400)
        H, W = shape_hw
        abs_pts[:, 0] = np.clip(abs_pts[:, 0], 0, W - 1)
        abs_pts[:, 1] = np.clip(abs_pts[:, 1], 0, H - 1)
        return abs_pts

    def _normalize_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        img = np.asarray(image, dtype=np.float64)
        finite = np.isfinite(img)
        if not finite.any():
            return np.zeros(img.shape, dtype=np.uint8)
        vmin = float(np.nanmin(img[finite]))
        vmax = float(np.nanmax(img[finite]))
        if vmax <= vmin:
            return np.zeros(img.shape, dtype=np.uint8)
        scaled = (img - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 255.0).round().astype(np.uint8)

    def _prompt_from_roi_state(
        self, state: dict, shape_hw: Tuple[int, int]
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        abs_pts = self._abs_pts_from_state_safe(state, shape_hw)
        if abs_pts.ndim != 2 or abs_pts.shape[0] < 3:
            return None, None
        H, W = shape_hw
        x = abs_pts[:, 0]
        y = abs_pts[:, 1]
        x1 = float(np.clip(np.min(x), 0, W - 1))
        x2 = float(np.clip(np.max(x), 0, W - 1))
        y1 = float(np.clip(np.min(y), 0, H - 1))
        y2 = float(np.clip(np.max(y), 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            return None, None
        cx = float(np.clip(np.mean(x), 0, W - 1))
        cy = float(np.clip(np.mean(y), 0, H - 1))
        return [x1, y1, x2, y2], [cx, cy]

    def _medsam2_debug_dir(self) -> Optional[str]:
        try:
            settings = get_medsam2_settings()
        except Exception:
            return None
        debug_dir = settings.get("debug_dir") or ""
        if debug_dir and settings.get("debug_keep"):
            return str(debug_dir)
        return None

    def _prompt_contour_point_count(self) -> Optional[int]:
        try:
            settings = get_medsam2_settings()
        except Exception:
            settings = {}
        default_points = int(settings.get("contour_points", 10))
        count, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Contour points",
            "Number of contour points:",
            default_points,
            3,
            200,
            1,
        )
        if not ok:
            return None
        return int(count)

    def _refine_pcmra_roi_for_phase(self, t: int, n_points: Optional[int] = None) -> Optional[str]:
        if t < 0 or t >= self.Nt:
            return "Invalid phase index."
        if self._is_roi_locked(t):
            return "ROI is locked."

        Ipcm, _, _, _, _ = self.reslice_for_phase(t)
        img_u8 = self._normalize_uint8(Ipcm)
        shape_hw = img_u8.shape[:2]

        st = self.roi_state[t]
        temp_state = st if st is not None else self.default_poly_roi_state(shape_hw)
        box, point = self._prompt_from_roi_state(temp_state, shape_hw)
        if box is None or point is None:
            return "Failed to build ROI prompt."

        if n_points is None:
            settings = get_medsam2_settings()
            n_points = int(settings.get("contour_points", 10))

        negative_points = []
        if self._negative_point_mode:
            negative_points = [list(pt) for pt in self._negative_points[t]]
        points_xy = [point] + negative_points
        point_labels = [1] + [0] * len(negative_points)

        self._begin_history_capture("roi", t)
        try:
            mask = run_medsam2_subprocess(img_u8, box, points_xy, point_labels)
            pts = mask_to_polygon_points(mask, n_points=int(n_points))
            if not pts:
                raise RuntimeError("Empty contour from MedSAM2 mask.")
            new_state = polygon_points_to_roi_state(pts, current_state=temp_state, shape_hw=shape_hw)
            self.roi_state[t] = new_state
            if t == int(self.slider.value()) - 1:
                self.apply_roi_state_both(new_state)
                self.update_spline_overlay(t)
                self.compute_current(update_only=True)
            self._commit_history_capture("roi", t)
            return None
        except Exception as exc:
            self._history_active = None
            return str(exc)

    def refine_roi_pcmra_phase(self) -> None:
        t = int(self.slider.value()) - 1
        n_points = self._prompt_contour_point_count()
        if n_points is None:
            return
        err = self._refine_pcmra_roi_for_phase(t, n_points=n_points)
        if err is not None:
            if "Empty contour" in err:
                debug_dir = self._medsam2_debug_dir()
                if debug_dir:
                    err += f"\nDebug artifacts: {debug_dir}"
            QtWidgets.QMessageBox.warning(self, "MV tracker", f"MedSAM2 refine failed:\n{err}")

    def refine_roi_pcmra_all_phases(self) -> None:
        n_points = self._prompt_contour_point_count()
        if n_points is None:
            return
        progress = QtWidgets.QProgressDialog(
            "Refining ROI with MedSAM2...", "Cancel", 0, self.Nt, self
        )
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)

        errors = []
        for t in range(self.Nt):
            progress.setValue(t)
            QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                break
            err = self._refine_pcmra_roi_for_phase(t, n_points=n_points)
            if err is not None and err != "ROI is locked.":
                errors.append(f"Phase {t + 1}: {err}")

        progress.setValue(self.Nt)
        if errors:
            msg = "Some phases failed to refine:\n" + "\n".join(errors[:10])
            if any("Empty contour" in err for err in errors):
                debug_dir = self._medsam2_debug_dir()
                if debug_dir:
                    msg += f"\nDebug artifacts: {debug_dir}"
            if len(errors) > 10:
                msg += f"\n... and {len(errors) - 10} more."
            QtWidgets.QMessageBox.warning(self, "MV tracker", msg)

    def _segment_flow_values(
        self,
        Vn: np.ndarray,
        spmm: float,
        mask: np.ndarray,
        center_xy: np.ndarray,
        ref_angle: float,
        n_segments: int,
    ) -> List[float]:
        if not np.any(mask):
            return [np.nan] * n_segments
        rows, cols = np.where(mask)
        dx = cols.astype(np.float64) - center_xy[0]
        dy = rows.astype(np.float64) - center_xy[1]
        seg_width = 2.0 * np.pi / float(n_segments)
        angles = (np.arctan2(-dy, dx) - ref_angle) % (2.0 * np.pi)
        seg_idx = np.floor(angles / seg_width).astype(int)
        seg_idx = np.clip(seg_idx, 0, n_segments - 1)

        dA_m2 = (spmm * 1e-3) ** 2
        values = []
        for seg in range(n_segments):
            sel = seg_idx == seg
            if not np.any(sel):
                values.append(np.nan)
                continue
            Q_m3s = float(np.nansum(Vn[rows[sel], cols[sel]]) * dA_m2)
            values.append(Q_m3s * 1e6)
        return self._rotate_segment_values(values, n_segments)

    def _segment_values_for_metric(
        self,
        metric: str,
        Ivelmag: np.ndarray,
        Vn: np.ndarray,
        Ike: Optional[np.ndarray],
        Ivort: Optional[np.ndarray],
        spmm: float,
        mask: np.ndarray,
        center_xy: np.ndarray,
        ref_angle: float,
        n_segments: int,
    ) -> List[float]:
        rows, cols = np.where(mask)
        if rows.size == 0:
            return [np.nan] * n_segments
        dx = cols.astype(np.float64) - center_xy[0]
        dy = rows.astype(np.float64) - center_xy[1]
        seg_width = 2.0 * np.pi / float(n_segments)
        angles = (np.arctan2(-dy, dx) - ref_angle) % (2.0 * np.pi)
        seg_idx = np.floor(angles / seg_width).astype(int)
        seg_idx = np.clip(seg_idx, 0, n_segments - 1)

        values = []
        for seg in range(n_segments):
            sel = seg_idx == seg
            if not np.any(sel):
                values.append(np.nan)
                continue
            rr = rows[sel]
            cc = cols[sel]
            if metric == "Flow rate (mL/s)":
                dA_m2 = (spmm * 1e-3) ** 2
                Q_m3s = float(np.nansum(Vn[rr, cc]) * dA_m2)
                values.append(Q_m3s * 1e6)
            elif metric == "Peak velocity (m/s)":
                values.append(float(np.nanmax(Ivelmag[rr, cc])))
            elif metric == "Mean velocity (m/s)":
                values.append(float(np.nanmean(Ivelmag[rr, cc])))
            elif metric == "Kinetic energy (uJ)":
                if Ike is None:
                    values.append(np.nan)
                else:
                    values.append(float(np.nansum(Ike[rr, cc]) * self._voxel_volume_m3 * 1e6))
            elif metric == "Peak vorticity (1/s)":
                if Ivort is None:
                    values.append(np.nan)
                else:
                    values.append(float(np.nanmax(Ivort[rr, cc])))
            elif metric == "Mean vorticity (1/s)":
                if Ivort is None:
                    values.append(np.nan)
                else:
                    values.append(float(np.nanmean(Ivort[rr, cc])))
            else:
                values.append(np.nan)
        return self._rotate_segment_values(values, n_segments)

    def _format_segments(self, values: Optional[List[float]]) -> str:
        if not values:
            return "-"
        out = []
        for idx, v in enumerate(values, start=1):
            val = f"{v:.2f}" if np.isfinite(v) else "-"
            out.append(f"R{idx}:{val}")
        return "[" + ", ".join(out) + "]"

    def _rotate_segment_values(self, values: List[float], n_segments: int) -> List[float]:
        if not values:
            return values
        shift = 1 if n_segments in (4, 6) else 0
        if shift == 0:
            return values
        shift = shift % n_segments
        return values[-shift:] + values[:-shift]

    def _set_segment_anchor(self, point_xy: np.ndarray, shape_hw: Tuple[int, int]) -> None:
        contour = self._roi_contour_points(shape_hw)
        if contour is None or contour.size == 0:
            return
        center = contour.mean(axis=0)
        diff = contour - point_xy[None, :]
        idx = int(np.argmin(np.sum(diff * diff, axis=1)))
        anchor = contour[idx]
        dx = anchor[0] - center[0]
        dy = anchor[1] - center[1]
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return
        self._segment_ref_angle[t] = float(np.arctan2(-dy, dx))
        self._segment_anchor_xy[t] = anchor.copy()
        self._segment_count[t] = 6 if self.segment_selector.currentText().startswith("6") else 4
        self.memo.appendPlainText(
            f"Segment anchor set at ({anchor[0]:.1f}, {anchor[1]:.1f}); regional sectors enabled."
        )
        if not self.chk_apply_segments.isChecked():
            self.chk_apply_segments.blockSignals(True)
            self.chk_apply_segments.setChecked(True)
            self.chk_apply_segments.blockSignals(False)
        self.compute_current(update_only=True)
        self._update_segment_overlay(t)

    def on_anchor_pick_pcmra(self, event):
        if event.button() != QtCore.Qt.MouseButton.RightButton:
            return
        view = self.pcmra_view.getView()
        img_item = self.pcmra_view.getImageItem()
        if img_item is None or img_item.image is None:
            return
        pos = view.mapSceneToView(event.scenePos())
        shape = img_item.image.shape
        if not (0 <= pos.x() < shape[1] and 0 <= pos.y() < shape[0]):
            return
        self._set_segment_anchor(np.array([pos.x(), pos.y()], dtype=np.float64), shape)

    def _update_segment_overlay(self, t: int):
        if not hasattr(self, "segment_curve_pcm") or self.segment_curve_pcm is None:
            return
        if not hasattr(self, "segment_curve_vel") or self.segment_curve_vel is None:
            return
        if hasattr(self, "segment_anchor_marker") and self.segment_anchor_marker is not None:
            anchor = self._segment_anchor_xy[t] if 0 <= t < self.Nt else None
            if anchor is None or not self._show_segments:
                self.segment_anchor_marker.setData([], [])
            else:
                self.segment_anchor_marker.setData(
                    [anchor[0]],
                    [anchor[1]],
                )
        if not self._show_segments or t < 0 or t >= self.Nt:
            self.segment_curve_pcm.setData([], [])
            self.segment_curve_vel.setData([], [])
            self._update_segment_labels(t)
            return
        ref_angle = self._segment_ref_angle[t]
        if ref_angle is None:
            self.segment_curve_pcm.setData([], [])
            self.segment_curve_vel.setData([], [])
            self._update_segment_labels(t)
            return
        st = self.roi_state[t] if 0 <= t < self.Nt else None
        if st is None:
            self.segment_curve_pcm.setData([], [])
            self.segment_curve_vel.setData([], [])
            self._update_segment_labels(t)
            return
        abs_pts = self._abs_pts_from_state_safe(st, (self.Npix, self.Npix))
        if abs_pts.ndim != 2 or abs_pts.shape[0] < 3:
            self.segment_curve_pcm.setData([], [])
            self.segment_curve_vel.setData([], [])
            self._update_segment_labels(t)
            return
        abs_pts = closed_spline_xy(abs_pts, n_out=400)
        center = abs_pts.mean(axis=0)
        r = np.max(np.linalg.norm(abs_pts - center[None, :], axis=1))
        if not np.isfinite(r) or r <= 0:
            self.segment_curve_pcm.setData([], [])
            self.segment_curve_vel.setData([], [])
            self._update_segment_labels(t)
            return
        count = self._segment_count[t] if 0 <= t < self.Nt else 6
        angles = [ref_angle + i * (2.0 * np.pi / count) for i in range(count)]
        xs = []
        ys = []
        for ang in angles:
            xs.extend([center[0], center[0] + r * np.cos(ang), np.nan])
            ys.extend([center[1], center[1] - r * np.sin(ang), np.nan])
        self.segment_curve_pcm.setData(np.array(xs), np.array(ys))
        self.segment_curve_vel.setData(np.array(xs), np.array(ys))
        self._update_segment_labels(t)

    def _update_segment_labels(self, t: int) -> None:
        if not self._segment_label_items_pcm or not self._segment_label_items_vel:
            return
        show = bool(self._show_segments and self.chk_segment_labels.isChecked())
        if not show or t < 0 or t >= self.Nt:
            for item in self._segment_label_items_pcm + self._segment_label_items_vel:
                item.setVisible(False)
            return
        ref_angle = self._segment_ref_angle[t]
        if ref_angle is None:
            for item in self._segment_label_items_pcm + self._segment_label_items_vel:
                item.setVisible(False)
            return
        contour = self._roi_contour_points((self.Npix, self.Npix))
        if contour is None or contour.ndim != 2 or contour.shape[0] < 3:
            for item in self._segment_label_items_pcm + self._segment_label_items_vel:
                item.setVisible(False)
            return
        center = contour.mean(axis=0)
        dist = np.linalg.norm(contour - center[None, :], axis=1)
        if dist.size == 0 or not np.isfinite(dist).any():
            for item in self._segment_label_items_pcm + self._segment_label_items_vel:
                item.setVisible(False)
            return
        radius = float(np.nanmedian(dist)) * 0.7
        count = self._segment_count[t] if 0 <= t < self.Nt else 6
        seg_width = 2.0 * np.pi / float(count)
        start_angle = ref_angle
        label_shift = 1 if count in (4, 6) else 0
        for idx in range(6):
            visible = idx < count and radius > 0
            label_idx = (idx + label_shift) % count
            label = f"R{label_idx + 1}"
            angle = start_angle + (idx + 0.5) * seg_width
            x = center[0] + radius * float(np.cos(angle))
            y = center[1] - radius * float(np.sin(angle))
            for item in (self._segment_label_items_pcm[idx], self._segment_label_items_vel[idx]):
                item.setVisible(visible)
                if visible:
                    item.setText(label)
                    item.setPos(x, y)

    def toggle_segments_visibility(self, _state: int):
        self._show_segments = bool(self.chk_apply_segments.isChecked())
        t = int(self.slider.value()) - 1
        if 0 <= t < self.Nt:
            self._segment_count[t] = 6 if self.segment_selector.currentText().startswith("6") else 4
        self._update_segment_overlay(int(self.slider.value()) - 1)
        self._update_segment_context_menu()
        self.compute_current(update_only=True)

    def _on_segment_labels_toggle(self, _state: int):
        self._update_segment_overlay(int(self.slider.value()) - 1)

    def _update_segment_context_menu(self):
        menu_enabled = not self._show_segments
        self.pcmra_view.getView().setMenuEnabled(menu_enabled)
        self.vel_view.getView().setMenuEnabled(menu_enabled)

    def _on_negative_points_toggle(self, _state: int) -> None:
        self._set_negative_point_mode(bool(self.chk_negative_points.isChecked()))

    def _set_negative_point_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._negative_point_mode = enabled
        self.chk_negative_points.blockSignals(True)
        try:
            self.chk_negative_points.setChecked(enabled)
        finally:
            self.chk_negative_points.blockSignals(False)
        if enabled:
            self.memo.appendPlainText("Negative point mode enabled (click on PCMRA to add points).")
        else:
            self.memo.appendPlainText("Negative point mode disabled.")
        if not self.brush_mode:
            cursor = QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor
            self.pcmra_view.getView().setCursor(cursor)

    def _add_negative_point(self, point_xy: np.ndarray, shape_hw: Tuple[int, int]) -> None:
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return
        x = float(np.clip(point_xy[0], 0, shape_hw[1] - 1))
        y = float(np.clip(point_xy[1], 0, shape_hw[0] - 1))
        self._negative_points[t].append([x, y])
        self._update_negative_point_overlay(t)
        self.memo.appendPlainText(f"Added negative point at ({x:.1f}, {y:.1f}).")

    def _remove_negative_point(
        self, point_xy: np.ndarray, shape_hw: Tuple[int, int], radius: float = 6.0
    ) -> bool:
        t = int(self.slider.value()) - 1
        if t < 0 or t >= self.Nt:
            return False
        pts = self._negative_points[t]
        if not pts:
            return False
        x = float(np.clip(point_xy[0], 0, shape_hw[1] - 1))
        y = float(np.clip(point_xy[1], 0, shape_hw[0] - 1))
        pts_arr = np.array(pts, dtype=np.float64)
        dists = np.linalg.norm(pts_arr - np.array([x, y], dtype=np.float64)[None, :], axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > radius:
            return False
        removed = pts.pop(idx)
        self._update_negative_point_overlay(t)
        self.memo.appendPlainText(f"Removed negative point at ({removed[0]:.1f}, {removed[1]:.1f}).")
        return True

    def _update_negative_point_overlay(self, t: int) -> None:
        if not hasattr(self, "_negative_point_marker") or self._negative_point_marker is None:
            return
        if t < 0 or t >= self.Nt:
            self._negative_point_marker.setData([], [])
            return
        pts = self._negative_points[t]
        if not pts:
            self._negative_point_marker.setData([], [])
            return
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        self._negative_point_marker.setData(xs, ys)

    def _on_fps_changed(self, value: float):
        self.play_fps = float(value)
        if self.btn_play.isChecked():
            self._play_timer.setInterval(int(round(1000.0 / max(self.play_fps, 1.0))))

    def toggle_playback(self, enabled: bool):
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(bool(enabled))
        self.btn_play.blockSignals(False)
        if enabled:
            self.btn_play.setText("Pause")
            self._play_timer.start(int(round(1000.0 / max(self.play_fps, 1.0))))
        else:
            self.btn_play.setText("Play")
            self._play_timer.stop()

    def _advance_playback(self):
        if not self.btn_play.isChecked():
            return
        next_val = self.slider.value() + 1
        if next_val > self.slider.maximum():
            next_val = self.slider.minimum()
        self.slider.setValue(next_val)

    def toggle_brush_mode(self):
        self.brush_mode = not self.brush_mode
        self.btn_brush.setText("Brush ROI: ON" if self.brush_mode else "Brush ROI: OFF")
        self.memo.appendPlainText(f"Brush mode {'enabled' if self.brush_mode else 'disabled'}.")
        if self.brush_mode:
            self._update_brush_cursor()
        else:
            cursor = (
                QtCore.Qt.CursorShape.CrossCursor
                if self._negative_point_mode
                else QtCore.Qt.CursorShape.ArrowCursor
            )
            self.pcmra_view.getView().setCursor(cursor)
            self.vel_view.getView().setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def _update_brush_cursor(self):
        diameter = int(max(6, self.brush_radius * 2))
        pix = QtGui.QPixmap(diameter, diameter)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = QtGui.QColor(0, 255, 255, 80)
        pen = QtGui.QPen(QtGui.QColor(0, 255, 255, 200))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(color))
        painter.drawEllipse(1, 1, diameter - 2, diameter - 2)
        painter.end()
        cursor = QtGui.QCursor(pix)
        self.pcmra_view.getView().setCursor(cursor)
        self.vel_view.getView().setCursor(cursor)

    def on_line_angle_changed(self, _value: float):
        self._cur_phase = None
        cur = int(self.slider.value()) - 1
        if 0 <= cur < len(self.line_angle):
            self.line_angle[cur] = float(self.spin_line_angle.value())
        self.set_phase(cur)
        self.compute_current(update_only=True)

    def _apply_brush_at(self, view: pg.ViewBox, pos: QtCore.QPointF):
        if not self.brush_mode:
            return
        t = int(self.slider.value()) - 1
        if self._is_roi_locked(t):
            return
        st = self.roi_state[t]
        if st is None:
            st = self.default_poly_roi_state((self.Npix, self.Npix))
            self.roi_state[t] = st
        abs_pts = self._abs_pts_from_state_safe(st, (self.Npix, self.Npix))
        if abs_pts.ndim != 2 or abs_pts.shape[0] < 3:
            return
        center = abs_pts.mean(axis=0)
        mask = polygon_mask((self.Npix, self.Npix), abs_pts)
        mouse = np.array([pos.x(), pos.y()], dtype=np.float64)
        dist = np.linalg.norm(abs_pts - mouse[None, :], axis=1)
        if not np.any(dist <= self.brush_radius):
            return
        in_mask = False
        mx, my = int(round(mouse[0])), int(round(mouse[1]))
        if 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1]:
            in_mask = bool(mask[my, mx])
        step = max(self.brush_radius * self.brush_strength, 0.2) * (1.0 if in_mask else -1.0)
        updated = abs_pts.copy()
        for i, d in enumerate(dist):
            if d > self.brush_radius:
                continue
            v = updated[i] - center
            vn = np.linalg.norm(v)
            if vn <= 1e-6:
                continue
            updated[i] = updated[i] + (v / vn) * step
        new_state = {
            "pos": (0.0, 0.0),
            "points": updated.tolist(),
            "closed": True,
        }
        self.roi_state[t] = new_state
        self.apply_roi_state_both(new_state)
        self.update_spline_overlay(t)
        self.compute_current(update_only=True)

    def eventFilter(self, obj, event):
        view = None
        if obj == self.pcmra_view.getView().scene():
            view = self.pcmra_view.getView()
        elif obj == self.vel_view.getView().scene():
            view = self.vel_view.getView()

        if view is not None and event.type() in (QtCore.QEvent.Type.Wheel, QtCore.QEvent.Type.GraphicsSceneWheel):
            if self.brush_mode and event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                if hasattr(event, "angleDelta"):
                    delta = 1 if event.angleDelta().y() > 0 else -1
                else:
                    delta = 1 if event.delta() > 0 else -1
                self.brush_radius = float(np.clip(self.brush_radius + delta * 2.0, 2.0, 100.0))
                self._update_brush_cursor()
                event.accept()
                return True

        if view is not None and event.type() in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.GraphicsSceneMousePress):
            if (
                view == self.pcmra_view.getView()
                and self._negative_point_mode
                and not self.brush_mode
                and event.button() == QtCore.Qt.MouseButton.LeftButton
            ):
                img_item = self.pcmra_view.getImageItem()
                if img_item is None or img_item.image is None:
                    return True
                if hasattr(event, "scenePos"):
                    pos = view.mapSceneToView(event.scenePos())
                else:
                    scene_pos = view.mapToScene(event.pos())
                    pos = view.mapSceneToView(scene_pos)
                shape = img_item.image.shape
                if 0 <= pos.x() < shape[1] and 0 <= pos.y() < shape[0]:
                    self._add_negative_point(np.array([pos.x(), pos.y()], dtype=np.float64), shape)
                event.accept()
                return True
            if (
                view == self.pcmra_view.getView()
                and self._negative_point_mode
                and not self.brush_mode
                and event.button() == QtCore.Qt.MouseButton.RightButton
            ):
                img_item = self.pcmra_view.getImageItem()
                if img_item is None or img_item.image is None:
                    return True
                if hasattr(event, "scenePos"):
                    pos = view.mapSceneToView(event.scenePos())
                else:
                    scene_pos = view.mapToScene(event.pos())
                    pos = view.mapSceneToView(scene_pos)
                shape = img_item.image.shape
                if 0 <= pos.x() < shape[1] and 0 <= pos.y() < shape[0]:
                    self._remove_negative_point(np.array([pos.x(), pos.y()], dtype=np.float64), shape)
                event.accept()
                return True
            if self.brush_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
                t = int(self.slider.value()) - 1
                self._begin_history_capture("roi", t)
                event.accept()
                return True

        if view is not None and event.type() in (QtCore.QEvent.Type.MouseButtonRelease, QtCore.QEvent.Type.GraphicsSceneMouseRelease):
            if self.brush_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
                t = int(self.slider.value()) - 1
                self._commit_history_capture("roi", t)
                event.accept()
                return True

        if view is not None and event.type() in (QtCore.QEvent.Type.MouseMove, QtCore.QEvent.Type.GraphicsSceneMouseMove):
            if self.brush_mode and QtWidgets.QApplication.mouseButtons() & QtCore.Qt.MouseButton.LeftButton:
                if hasattr(event, "scenePos"):
                    pos = view.mapSceneToView(event.scenePos())
                else:
                    scene_pos = view.mapToScene(event.pos())
                    pos = view.mapSceneToView(scene_pos)
                self._apply_brush_at(view, pos)
                return True

        return super().eventFilter(obj, event)

    # ============================
    # Missing helpers
    # ============================
    def _store_view_range(self, key: str):
        if self._restoring_view or self._updating_image or self._syncing_view:
            return
        view = self._view_for_key(key)
        img_view = self._image_view_for_key(key)
        img_item = img_view.getImageItem()
        if img_item is None or img_item.image is None:
            return
        rng = view.viewRange()
        self._view_ranges[key] = rng
        if key in ("pcmra", "vel"):
            other_key = "vel" if key == "pcmra" else "pcmra"
            other_view = self._view_for_key(other_key)
            self._syncing_view = True
            try:
                other_view.setRange(xRange=rng[0], yRange=rng[1], padding=0.0)
            finally:
                self._syncing_view = False
            self._view_ranges[other_key] = rng

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
        flip = -1.0 if self.chk_flip_flow.isChecked() else 1.0
        Q = self.metrics_Q[t] * flip
        Vpk = self.metrics_Vpk[t]
        Vmn = self.metrics_Vmn[t]
        KE = self.metrics_KE[t]
        VortPk = self.metrics_VortPk[t]
        VortMn = self.metrics_VortMn[t]
        seg_metric = self.chart_selector.currentText()
        seg_count = self._segment_count[t] if 0 <= t < self.Nt else 6
        seg4 = self.metrics_seg4.get(seg_metric, [None] * self.Nt)[t]
        seg6 = self.metrics_seg6.get(seg_metric, [None] * self.Nt)[t]
        self.lbl_Q.setText(f"Flow rate (mL/s): {Q:.3f}" if np.isfinite(Q) else "Flow rate (mL/s): -")
        seg4_txt = self._format_segments(seg4)
        seg6_txt = self._format_segments(seg6)
        if seg4 or seg6:
            seg_vals = seg6_txt if seg_count == 6 else seg4_txt
            self.lbl_segments.setText(f"Segments R1..: {seg_vals}")
        else:
            self.lbl_segments.setText("Segments: -")
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
        flip = -1.0 if self.chk_flip_flow.isChecked() else 1.0
        if label == "Flow rate (mL/s)":
            self.plot.plot_metric(phases, self.metrics_Q * flip, label, "tab:blue")
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

        if self.chk_plot_segments.isChecked():
            seg_count = 6 if self.segment_selector.currentText().startswith("6") else 4
            seg_values = self.metrics_seg6.get(label) if seg_count == 6 else self.metrics_seg4.get(label)
            if seg_values:
                colors = ["tab:cyan", "tab:olive", "tab:pink", "tab:gray", "tab:green", "tab:orange"]
                for idx in range(seg_count):
                    series = []
                    for t in range(self.Nt):
                        vals = seg_values[t]
                        series.append(vals[idx] if vals is not None and idx < len(vals) else np.nan)
                    if label == "Flow rate (mL/s)":
                        series = np.array(series) * flip
                    self.plot.ax.plot(
                        phases,
                        series,
                        marker="o",
                        color=colors[idx % len(colors)],
                        label=f"R{idx + 1}",
                        linestyle="--",
                    )
                self.plot.ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
                try:
                    self.plot.fig.tight_layout(rect=[0, 0, 0.8, 1])
                except Exception:
                    pass
                self.plot.draw()
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
        if not hasattr(self, "btn_roi_lock"):
            return
        t = int(self.slider.value()) - 1
        locked = self._is_roi_locked(t)
        self.btn_roi_lock.setText("Lock ROI: ON" if locked else "Lock ROI: OFF")
        self._update_lock_label_visibility()

    def toggle_roi_lock(self):
        if not hasattr(self, "btn_roi_lock"):
            return
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
        segment_payload = {}
        if self.chk_plot_segments.isChecked():
            segment_payload = {"seg4": self.metrics_seg4, "seg6": self.metrics_seg6}
        save_tracking_state_h5(
            path=out_path,
            line_norm=self.line_norm,
            line_angle=self.line_angle,
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
            segment_payload=segment_payload,
            segment_count=6 if self.segment_selector.currentText().startswith("6") else 4,
            plot_segments=self.chk_plot_segments.isChecked(),
            segment_ref_angle=self._segment_ref_angle,
            segment_anchor_xy=self._segment_anchor_xy,
            segment_count_list=self._segment_count,
            apply_segments=self.chk_apply_segments.isChecked(),
            show_segment_labels=self.chk_segment_labels.isChecked(),
            flip_flow=self.chk_flip_flow.isChecked(),
            axis_order=self.axis_order,
            axis_flips=self.axis_flips,
        )
        self.tracking_path = out_path
        self.memo.appendPlainText(f"Saved tracking state to: {out_path}")

    def _metric_labels(self) -> List[str]:
        return [
            "Flow rate (mL/s)",
            "Peak velocity (m/s)",
            "Mean velocity (m/s)",
            "Kinetic energy (uJ)",
            "Peak vorticity (1/s)",
            "Mean vorticity (1/s)",
        ]

    def _format_value(self, value: float) -> str:
        return f"{value:.6f}" if np.isfinite(value) else ""

    def copy_data_to_clipboard(self):
        header = ["Phase"] + self._metric_labels()
        lines = ["\t".join(header)]
        flip = -1.0 if self.chk_flip_flow.isChecked() else 1.0
        for t in range(self.Nt):
            row = [
                f"{t + 1}",
                self._format_value(self.metrics_Q[t] * flip),
                self._format_value(self.metrics_Vpk[t]),
                self._format_value(self.metrics_Vmn[t]),
                self._format_value(self.metrics_KE[t]),
                self._format_value(self.metrics_VortPk[t]),
                self._format_value(self.metrics_VortMn[t]),
            ]
            lines.append("\t".join(row))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.memo.appendPlainText("Copied contour metrics to clipboard.")

    def _segment_values_for_phase(self, metric: str, t: int, seg_count: int) -> List[float]:
        seg_values = self.metrics_seg6.get(metric) if seg_count == 6 else self.metrics_seg4.get(metric)
        if not seg_values or t >= len(seg_values) or seg_values[t] is None:
            return [np.nan] * seg_count
        return [seg_values[t][idx] if idx < len(seg_values[t]) else np.nan for idx in range(seg_count)]

    def copy_regional_data_to_clipboard(self):
        seg_count = 6 if self.segment_selector.currentText().startswith("6") else 4
        selection = self.copy_regional_selector.currentText()
        flip = -1.0 if self.chk_flip_flow.isChecked() else 1.0
        lines = []
        if selection == "Current chart":
            metric = self.chart_selector.currentText()
            header = ["Phase"] + [f"R{idx + 1} ({metric})" for idx in range(seg_count)]
            lines.append("\t".join(header))
            for t in range(self.Nt):
                values = self._segment_values_for_phase(metric, t, seg_count)
                if metric == "Flow rate (mL/s)":
                    values = [value * flip for value in values]
                row = [f"{t + 1}"] + [self._format_value(v) for v in values]
                lines.append("\t".join(row))
        else:
            metrics = self._metric_labels()
            header = ["Phase"]
            for metric in metrics:
                header.extend([f"R{idx + 1} ({metric})" for idx in range(seg_count)])
            lines.append("\t".join(header))
            for t in range(self.Nt):
                row = [f"{t + 1}"]
                for metric in metrics:
                    values = self._segment_values_for_phase(metric, t, seg_count)
                    if metric == "Flow rate (mL/s)":
                        values = [value * flip for value in values]
                    row.extend([self._format_value(v) for v in values])
                lines.append("\t".join(row))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.memo.appendPlainText("Copied regional metrics to clipboard.")

    def copy_current_to_clipboard(self):
        self.copy_data_to_clipboard()

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
