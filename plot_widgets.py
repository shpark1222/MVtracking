import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph.opengl as gl

import matplotlib

matplotlib.use("QtAgg")
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._phases = None
        self._values = None
        self._phase_line = None
        self._phase_callback = None
        super().__init__(self.fig)
        self.setParent(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def plot_metric(self, phases, values, label: str, color: str):
        self.ax.clear()
        self.ax.grid(True)
        self._phases = phases
        self._values = values
        self.ax.plot(phases, values, marker="o", color=color, label=label)
        self.ax.set_xlabel("Phase")
        self.ax.set_ylabel(label)
        self.ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.0)
        try:
            self.fig.tight_layout(rect=[0, 0, 1.0, 1])
        except Exception:
            pass
        if self._phase_line is not None:
            self._phase_line = None
        self.draw()

    def set_phase_indicator(self, phase: int):
        if phase is None:
            return
        if self._phase_line is not None:
            try:
                self._phase_line.remove()
            except Exception:
                pass
            self._phase_line = None
        self._phase_line = self.ax.axvline(float(phase), color="k", linestyle="--", linewidth=1)
        self.draw()

    def set_phase_callback(self, callback):
        self._phase_callback = callback

    def _on_click(self, event):
        if event.inaxes != self.ax or self._phases is None:
            return
        if event.xdata is None:
            return
        phases = np.array(self._phases, dtype=float)
        idx = int(np.argmin(np.abs(phases - event.xdata)))
        phase = int(phases[idx])
        if self._phase_callback is not None:
            self._phase_callback(phase)

    def _step_phase(self, step: int):
        if self._phase_callback is None or self._phases is None:
            return
        phases = np.array(self._phases, dtype=int)
        if phases.size == 0:
            return
        current = int(phases[0])
        if self._phase_line is not None:
            current = int(round(self._phase_line.get_xdata()[0]))
        idx = int(np.argmin(np.abs(phases - current)))
        idx = int(np.clip(idx + step, 0, len(phases) - 1))
        self._phase_callback(int(phases[idx]))

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        step = 1 if delta > 0 else -1
        self._step_phase(step)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Down):
            self._step_phase(-1)
            return
        if event.key() in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Up):
            self._step_phase(1)
            return
        super().keyPressEvent(event)


class _WheelToSliderFilter(QtCore.QObject):
    def __init__(self, slider: QtWidgets.QSlider):
        super().__init__()
        self.slider = slider

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Wheel:
            dy = event.angleDelta().y()
            if dy == 0:
                return False
            step = 1 if dy > 0 else -1
            v = self.slider.value() + step
            v = max(self.slider.minimum(), min(self.slider.maximum(), v))
            if v != self.slider.value():
                self.slider.setValue(v)
            event.accept()
            return True
        return False


class SyncableGLViewWidget(gl.GLViewWidget):
    cameraChanged = QtCore.Signal()

    def _emit_camera_changed(self):
        self.cameraChanged.emit()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self._emit_camera_changed()

    def wheelEvent(self, ev):
        super().wheelEvent(ev)
        self._emit_camera_changed()

    def keyReleaseEvent(self, ev):
        super().keyReleaseEvent(ev)
        self._emit_camera_changed()


class StreamlineWindow(QtWidgets.QWidget):
    camera_changed = QtCore.Signal(object)

    def __init__(
        self,
        axis_order: str = "XYZ",
        axis_flips: tuple = (False, False, False),
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Streamline View")
        self.axis_order = str(axis_order).upper()
        self.axis_flips = tuple(axis_flips)
        self._line_items = []
        self._streamlines = []
        self._volume_shape = None
        self._contour_item = None
        self._contour_points = None
        self._suppress_camera_signal = False

        self._build_view()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.view)

    def _build_view(self):
        self.view = SyncableGLViewWidget()
        self.view.opts["distance"] = 200
        self.view.setBackgroundColor("k")
        self.view.cameraChanged.connect(self._on_view_camera_changed)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._ensure_view():
            self.view.update()

    def _ensure_view(self):
        needs_rebuild = False
        if self.view is None:
            needs_rebuild = True
        else:
            context = None
            try:
                context = self.view.context()
            except Exception:
                context = None
            if context is None or not context.isValid():
                needs_rebuild = True
        if not needs_rebuild:
            return False
        layout = self.layout()
        if self.view is not None:
            try:
                self.view.setParent(None)
            except Exception:
                pass
        self._build_view()
        if layout is not None:
            layout.addWidget(self.view)
        if self._streamlines and self._volume_shape is not None:
            self.update_streamlines(self._streamlines, self._volume_shape)
        if self._contour_points is not None and self._volume_shape is not None:
            self.update_contour(self._contour_points, self._volume_shape)
        return True

    def _on_view_camera_changed(self):
        if self._suppress_camera_signal:
            return
        state = self.camera_state()
        if state is None:
            return
        if hasattr(self, "camera_changed"):
            try:
                self.camera_changed.emit(state)
            except Exception:
                pass

    def camera_state(self):
        if self.view is None:
            return None
        keys = ["center", "distance", "azimuth", "elevation", "fov"]
        return {key: self.view.opts.get(key) for key in keys if key in self.view.opts}

    def set_camera_state(self, state):
        if self.view is None or state is None:
            return
        self._suppress_camera_signal = True
        try:
            for key, value in state.items():
                if key in self.view.opts:
                    self.view.opts[key] = value
            self.view.update()
        finally:
            self._suppress_camera_signal = False

    def clear_streamlines(self):
        for item in self._line_items:
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self._line_items = []

    def clear_contour(self):
        if self._contour_item is None:
            return
        try:
            self.view.removeItem(self._contour_item)
        except Exception:
            pass
        self._contour_item = None

    def update_streamlines(self, streamlines, volume_shape):
        self.clear_streamlines()
        self._volume_shape = volume_shape
        self._streamlines = list(streamlines) if streamlines is not None else []
        if not self._streamlines:
            return
        self._update_view_center(volume_shape)
        prepared = []
        magnitudes = []
        for entry in self._streamlines:
            if entry is None:
                continue
            mags = None
            line = entry
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                line, mags = entry
            if line is None or len(line) == 0:
                continue
            pts = np.asarray(line, dtype=np.float32)
            mags_arr = None
            if mags is not None:
                mags_arr = np.asarray(mags, dtype=np.float32)
                if mags_arr.shape[0] != pts.shape[0]:
                    mags_arr = None
                elif np.any(~np.isfinite(mags_arr)):
                    mags_arr = None
            if mags_arr is not None:
                magnitudes.append(mags_arr)
            prepared.append((pts, mags_arr))

        mag_min = None
        mag_max = None
        if magnitudes:
            all_mags = np.concatenate(magnitudes)
            if all_mags.size > 0 and np.all(np.isfinite(all_mags)):
                mag_min = float(np.min(all_mags))
                mag_max = float(np.max(all_mags))

        for line, mags in prepared:
            pts = self._transform_points(line, volume_shape)
            colors = self._streamline_colors(pts.shape[0], mags, mag_min, mag_max)
            item = gl.GLLinePlotItem(
                pos=pts,
                color=colors,
                width=1.0,
                antialias=True,
                mode="line_strip",
            )
            self.view.addItem(item)
            self._line_items.append(item)

    def update_contour(self, contour_points, volume_shape):
        self.clear_contour()
        self._volume_shape = volume_shape
        self._contour_points = None if contour_points is None else np.asarray(contour_points, dtype=np.float32)
        if self._contour_points is None or self._contour_points.size == 0:
            return
        if self._contour_points.ndim != 2 or self._contour_points.shape[1] != 3:
            return
        pts = self._transform_points(self._contour_points, volume_shape)
        if pts.shape[0] < 2:
            return
        pts = np.vstack([pts, pts[0]])
        self._update_view_center(volume_shape)
        self._contour_item = gl.GLLinePlotItem(
            pos=pts,
            color=(1.0, 1.0, 1.0, 0.9),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.view.addItem(self._contour_item)

    def _update_view_center(self, volume_shape):
        if volume_shape is None:
            return
        shape_xyz = np.array([volume_shape[1], volume_shape[0], volume_shape[2]], dtype=float)
        center = shape_xyz / 2.0
        self.view.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))

    def _transform_points(self, points: np.ndarray, volume_shape):
        if points.size == 0:
            return points
        return points[:, [1, 0, 2]].astype(np.float32)

    def _streamline_colors(self, count: int, magnitudes=None, vmin=None, vmax=None) -> np.ndarray:
        if count <= 1:
            return np.array([[0.0, 0.0, 0.5, 0.9]], dtype=np.float32)
        cmap = cm.get_cmap("jet")
        if magnitudes is None or vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
            positions = np.linspace(0.0, 1.0, count, dtype=np.float32)
            colors = cmap(positions)
        else:
            denom = vmax - vmin
            if abs(denom) < 1e-8:
                positions = np.zeros(count, dtype=np.float32)
            else:
                positions = (magnitudes - vmin) / denom
                positions = np.clip(positions, 0.0, 1.0)
            colors = cmap(positions)
        colors[:, 3] = 0.9
        return colors.astype(np.float32)


class StreamlineGalleryWindow(QtWidgets.QWidget):
    seed_requested = QtCore.Signal(int)
    open_single_view = QtCore.Signal()

    def __init__(self, axis_orders, axis_flips, parent=None, columns: int = 4):
        super().__init__(parent)
        self.setWindowTitle("Streamline Gallery")
        self.views = []
        self._syncing_camera = False
        self._seed_phase = None
        self._seed_points = None

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        row = 0
        col = 0
        for order in axis_orders:
            for flips in axis_flips:
                panel = QtWidgets.QWidget()
                panel_layout = QtWidgets.QVBoxLayout(panel)
                panel_layout.setContentsMargins(4, 4, 4, 4)
                label = QtWidgets.QLabel(f"{order} | flips={flips}")
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                view = StreamlineWindow(axis_order=order, axis_flips=flips, parent=panel)
                view.setMinimumSize(240, 240)
                view.camera_changed.connect(
                    lambda state, source=view: self._sync_camera(source, state)
                )
                panel_layout.addWidget(label)
                panel_layout.addWidget(view)
                grid.addWidget(panel, row, col)
                self.views.append((view, order, flips))
                col += 1
                if col >= columns:
                    col = 0
                    row += 1

        scroll.setWidget(container)
        layout = QtWidgets.QVBoxLayout(self)
        control_row = QtWidgets.QHBoxLayout()
        control_row.addWidget(QtWidgets.QLabel("Seed count"))
        self.seed_spin = QtWidgets.QSpinBox(self)
        self.seed_spin.setRange(1, 5000)
        self.seed_spin.setValue(200)
        control_row.addWidget(self.seed_spin)
        seed_btn = QtWidgets.QPushButton("Seed from contour", self)
        seed_btn.clicked.connect(lambda: self.seed_requested.emit(self.seed_spin.value()))
        control_row.addWidget(seed_btn)
        open_btn = QtWidgets.QPushButton("Open single view tab", self)
        open_btn.clicked.connect(self.open_single_view.emit)
        control_row.addWidget(open_btn)
        control_row.addStretch(1)
        layout.addLayout(control_row)
        layout.addWidget(scroll)

    def set_seed_points(self, seed_points, phase: int):
        self._seed_points = None if seed_points is None else np.asarray(seed_points, dtype=np.float32)
        self._seed_phase = phase

    def seed_points_for_phase(self, phase: int):
        if self._seed_points is None or self._seed_phase != phase:
            return None
        return self._seed_points

    def _sync_camera(self, source_view, state):
        if self._syncing_camera:
            return
        self._syncing_camera = True
        try:
            for view, _order, _flips in self.views:
                if view is source_view:
                    continue
                view.set_camera_state(state)
        finally:
            self._syncing_camera = False


class StreamlinePlayerWindow(QtWidgets.QWidget):
    phase_changed = QtCore.Signal(int)
    seed_requested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Streamline Player")
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance_phase)
        self._total_steps = 0
        self._step_index = 0
        self._start_phase = 1
        self._phase = 1
        self._use_contour_seed = False

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Axis order"))
        self.axis_order_combo = QtWidgets.QComboBox(self)
        self.axis_order_combo.addItems(["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
        self.axis_order_combo.currentTextChanged.connect(lambda _val: self.phase_changed.emit(self._phase))
        controls.addWidget(self.axis_order_combo)
        controls.addWidget(QtWidgets.QLabel("Axis flips"))
        self.axis_flip_combo = QtWidgets.QComboBox(self)
        self.axis_flip_combo.addItems(
            [
                "None",
                "Flip X",
                "Flip Y",
                "Flip Z",
                "Flip X,Y",
                "Flip X,Z",
                "Flip Y,Z",
                "Flip X,Y,Z",
            ]
        )
        self.axis_flip_combo.currentTextChanged.connect(lambda _val: self.phase_changed.emit(self._phase))
        controls.addWidget(self.axis_flip_combo)
        controls.addWidget(QtWidgets.QLabel("Start phase"))
        self.start_phase_spin = QtWidgets.QSpinBox(self)
        self.start_phase_spin.setRange(1, 1)
        controls.addWidget(self.start_phase_spin)
        controls.addWidget(QtWidgets.QLabel("Cycles"))
        self.cycle_spin = QtWidgets.QSpinBox(self)
        self.cycle_spin.setRange(1, 20)
        self.cycle_spin.setValue(1)
        controls.addWidget(self.cycle_spin)
        controls.addWidget(QtWidgets.QLabel("Interval (ms)"))
        self.interval_spin = QtWidgets.QSpinBox(self)
        self.interval_spin.setRange(10, 5000)
        self.interval_spin.setValue(150)
        controls.addWidget(self.interval_spin)
        play_btn = QtWidgets.QPushButton("Play", self)
        play_btn.clicked.connect(self.start_playback)
        controls.addWidget(play_btn)
        stop_btn = QtWidgets.QPushButton("Stop", self)
        stop_btn.clicked.connect(self.stop_playback)
        controls.addWidget(stop_btn)
        layout.addLayout(controls)

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(QtWidgets.QLabel("Seed count"))
        self.seed_spin = QtWidgets.QSpinBox(self)
        self.seed_spin.setRange(1, 5000)
        self.seed_spin.setValue(200)
        seed_row.addWidget(self.seed_spin)
        seed_btn = QtWidgets.QPushButton("Seed from contour", self)
        seed_btn.clicked.connect(self._on_seed_clicked)
        seed_row.addWidget(seed_btn)
        seed_row.addStretch(1)
        self.phase_label = QtWidgets.QLabel("Phase: -")
        seed_row.addWidget(self.phase_label)
        layout.addLayout(seed_row)

        self.view = StreamlineWindow(parent=self)
        layout.addWidget(self.view)

    def set_phase_count(self, count: int) -> None:
        count = max(1, int(count))
        self.start_phase_spin.setRange(1, count)

    def axis_order(self) -> str:
        return self.axis_order_combo.currentText()

    def axis_flips(self) -> tuple:
        flip_options = [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ]
        return flip_options[self.axis_flip_combo.currentIndex()]

    def use_contour_seed(self) -> bool:
        return self._use_contour_seed

    def set_phase(self, phase: int) -> None:
        self._phase = int(phase)
        self.phase_label.setText(f"Phase: {self._phase}")
        self.phase_changed.emit(self._phase)

    def start_playback(self) -> None:
        self._start_phase = int(self.start_phase_spin.value())
        cycles = int(self.cycle_spin.value())
        self._total_steps = max(1, cycles) * max(1, self.start_phase_spin.maximum())
        self._step_index = 0
        self._timer.start(int(self.interval_spin.value()))
        self.set_phase(self._start_phase)

    def stop_playback(self) -> None:
        self._timer.stop()

    def _advance_phase(self) -> None:
        phase_count = int(self.start_phase_spin.maximum())
        if phase_count <= 0:
            return
        self._step_index += 1
        if self._step_index >= self._total_steps:
            self.stop_playback()
            return
        phase = ((self._start_phase - 1 + self._step_index) % phase_count) + 1
        self.set_phase(phase)

    def _on_seed_clicked(self) -> None:
        self._use_contour_seed = True
        self.seed_requested.emit(int(self.seed_spin.value()))


class StreamlineTabbedWindow(QtWidgets.QWidget):
    def __init__(self, axis_orders, axis_flips, parent=None, columns: int = 4):
        super().__init__(parent)
        self.setWindowTitle("Streamline Viewer")
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabs)

        self.gallery = StreamlineGalleryWindow(axis_orders, axis_flips, parent=self, columns=columns)
        self.player = StreamlinePlayerWindow(parent=self)
        self.tabs.addTab(self.gallery, "Gallery view")
        self.tabs.addTab(self.player, "Single view")

        self.gallery.open_single_view.connect(self._show_single_view_tab)

    def _show_single_view_tab(self) -> None:
        self.tabs.setCurrentWidget(self.player)
