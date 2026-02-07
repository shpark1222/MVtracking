import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from typing import Optional

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
        self._iso_item = None
        self._iso_volume = None
        self._iso_threshold = 0.0
        self._iso_enabled = False
        self._mask_iso_item = None
        self._mask_iso_volume = None
        self._mask_iso_enabled = False
        self._suppress_camera_signal = False
        self._prepared_streamlines = []
        self._show_streamlines = True
        self._particles_enabled = False
        self._particle_item = None
        self._particle_timer = QtCore.QTimer(self)
        self._particle_timer.timeout.connect(self._advance_particles)
        self._particle_step = 0
        self._particle_cycles = 1
        self._particle_cycle_index = 0
        self._particle_tracks = []
        self._particle_tracks_source = None
        self._pathline_items = []
        self._show_pathlines = False
        self._rebuilding_view = False

        self._build_view()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.view)

    def _build_view(self):
        self.view = SyncableGLViewWidget(self)
        self.view.opts["distance"] = 200
        self.view.setBackgroundColor("k")
        self.view.cameraChanged.connect(self._on_view_camera_changed)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._ensure_view():
            self.view.update()

    def _ensure_view(self):
        if self._rebuilding_view:
            return False
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
        self._particle_timer.stop()
        self._line_items = []
        self._pathline_items = []
        self._particle_item = None
        self._contour_item = None
        self._iso_item = None
        self._mask_iso_item = None
        self._rebuilding_view = True
        try:
            self._build_view()
            if layout is not None:
                layout.addWidget(self.view)
            if self._streamlines and self._volume_shape is not None:
                self.update_streamlines(self._streamlines, self._volume_shape, ensure_view=False)
            if self._contour_points is not None and self._volume_shape is not None:
                self.update_contour(self._contour_points, self._volume_shape)
            if self._iso_volume is not None and self._iso_enabled:
                self.update_isosurface(self._iso_volume, self._iso_threshold, enabled=True)
            if self._mask_iso_volume is not None and self._mask_iso_enabled:
                self.update_mask_isosurface(self._mask_iso_volume, enabled=True)
            if self._particles_enabled and self._particle_tracks:
                self._update_particles(self._particle_tracks)
        finally:
            self._rebuilding_view = False
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

    def clear_streamlines(self, clear_prepared: bool = True):
        for item in self._line_items:
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self._line_items = []
        if clear_prepared:
            self._prepared_streamlines = []

    def clear_particle_tracks(self) -> None:
        self._particle_tracks = []
        self._particle_tracks_source = None
        self.clear_particles()

    def clear_particles(self):
        if self._particle_item is None:
            return
        try:
            self.view.removeItem(self._particle_item)
        except Exception:
            pass
        self._particle_item = None
        self._particle_timer.stop()
        self._particle_step = 0
        self._particle_cycle_index = 0
        self._clear_pathlines()

    def clear_contour(self):
        if self._contour_item is None:
            return
        try:
            self.view.removeItem(self._contour_item)
        except Exception:
            pass
        self._contour_item = None

    def clear_isosurface(self):
        if self._iso_item is None:
            return
        try:
            self.view.removeItem(self._iso_item)
        except Exception:
            pass
        self._iso_item = None

    def clear_mask_isosurface(self):
        if self._mask_iso_item is None:
            return
        try:
            self.view.removeItem(self._mask_iso_item)
        except Exception:
            pass
        self._mask_iso_item = None

    def update_streamlines(self, streamlines, volume_shape, ensure_view: bool = True):
        if ensure_view:
            self._ensure_view()
        self.clear_streamlines()
        self._volume_shape = volume_shape
        self._streamlines = list(streamlines) if streamlines is not None else []
        self._particle_step = 0
        self._particle_cycle_index = 0
        if not self._streamlines:
            self._update_particles([])
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
            if pts.ndim != 2 or pts.shape[1] != 3:
                continue
            if not np.all(np.isfinite(pts)):
                valid_mask = np.all(np.isfinite(pts), axis=1)
                if not np.any(valid_mask):
                    continue
                first_invalid = int(np.argmax(~valid_mask)) if not np.all(valid_mask) else pts.shape[0]
                pts = pts[:first_invalid]
                if pts.size == 0:
                    continue
            mags_arr = None
            if mags is not None:
                mags_arr = np.asarray(mags, dtype=np.float32)
                if mags_arr.shape[0] != pts.shape[0]:
                    if mags_arr.shape[0] >= pts.shape[0]:
                        mags_arr = mags_arr[: pts.shape[0]]
                    else:
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
            self._prepared_streamlines.append((pts, colors))
        if self._show_streamlines:
            for pts, colors in self._prepared_streamlines:
                item = gl.GLLinePlotItem(
                    pos=pts,
                    color=colors,
                    width=1.0,
                    antialias=True,
                    mode="line_strip",
                )
                self.view.addItem(item)
                self._line_items.append(item)
        if self._particle_tracks_source != "precomputed":
            self._particle_tracks = list(self._prepared_streamlines)
            self._particle_tracks_source = "streamlines"
        self._update_particles(self._particle_tracks)

    def update_particle_tracks(self, tracks, volume_shape, ensure_view: bool = True):
        if ensure_view:
            self._ensure_view()
        self._volume_shape = volume_shape
        self._particle_tracks = []
        self._particle_tracks_source = "precomputed"
        if tracks is None:
            self._update_particles(self._particle_tracks)
            return
        prepared = []
        magnitudes = []
        for entry in tracks:
            if entry is None:
                continue
            mags = None
            line = entry
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                line, mags = entry
            if line is None or len(line) == 0:
                continue
            pts = np.asarray(line, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 3:
                continue
            if not np.all(np.isfinite(pts)):
                valid_mask = np.all(np.isfinite(pts), axis=1)
                if not np.any(valid_mask):
                    continue
                first_invalid = int(np.argmax(~valid_mask)) if not np.all(valid_mask) else pts.shape[0]
                pts = pts[:first_invalid]
                if pts.size == 0:
                    continue
            mags_arr = None
            if mags is not None:
                mags_arr = np.asarray(mags, dtype=np.float32)
                if mags_arr.shape[0] != pts.shape[0]:
                    if mags_arr.shape[0] >= pts.shape[0]:
                        mags_arr = mags_arr[: pts.shape[0]]
                    else:
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
            self._particle_tracks.append((pts, colors))
        self._particle_step = 0
        self._particle_cycle_index = 0
        self._update_particles(self._particle_tracks)

    def set_show_streamlines(self, show: bool) -> None:
        show = bool(show)
        if self._show_streamlines == show:
            return
        self._show_streamlines = show
        self._ensure_view()
        self.clear_streamlines(clear_prepared=False)
        if self._show_streamlines and self._prepared_streamlines:
            for pts, colors in self._prepared_streamlines:
                item = gl.GLLinePlotItem(
                    pos=pts,
                    color=colors,
                    width=1.0,
                    antialias=True,
                    mode="line_strip",
                )
                self.view.addItem(item)
                self._line_items.append(item)

    def set_show_pathlines(self, show: bool) -> None:
        show = bool(show)
        if self._show_pathlines == show:
            return
        self._show_pathlines = show
        if not self._show_pathlines:
            self._clear_pathlines()
        else:
            self._update_particles(self._particle_tracks)

    def set_particles_enabled(self, enabled: bool, interval_ms: int = 50) -> None:
        enabled = bool(enabled)
        if self._particles_enabled == enabled:
            if enabled and interval_ms and interval_ms > 0:
                self._particle_timer.setInterval(int(interval_ms))
            return
        self._particles_enabled = enabled
        if not enabled:
            self.clear_particles()
            return
        self._ensure_view()
        if self._particle_item is None:
            self._particle_item = gl.GLScatterPlotItem(size=4.0, pxMode=True)
            self.view.addItem(self._particle_item)
        if interval_ms and interval_ms > 0:
            self._particle_timer.setInterval(int(interval_ms))
            self._particle_timer.start()
        else:
            self._particle_timer.stop()
        self._update_particles(self._particle_tracks)

    def pause_particle_animation(self) -> None:
        self._particle_timer.stop()

    def reset_particle_animation(self) -> None:
        self._particle_step = 0
        self._particle_cycle_index = 0
        if self._particles_enabled:
            self._update_particles(self._particle_tracks)

    def set_particle_cycles(self, cycles: int) -> None:
        cycles = max(1, int(cycles))
        if self._particle_cycles == cycles:
            return
        self._particle_cycles = cycles
        self._particle_cycle_index = 0
        self._particle_step = 0

    def _update_particles(self, prepared_streamlines) -> None:
        if not self._particles_enabled:
            return
        if self._particle_item is None:
            self._particle_item = gl.GLScatterPlotItem(size=4.0, pxMode=True)
            self.view.addItem(self._particle_item)
        positions = []
        colors = []
        for idx_stream, (pts, stream_colors) in enumerate(prepared_streamlines):
            if pts.size == 0:
                continue
            if self._particle_step >= pts.shape[0]:
                continue
            idx = min(self._particle_step, pts.shape[0] - 1)
            pos = pts[idx]
            if not np.all(np.isfinite(pos)):
                continue
            positions.append(pos)
            if stream_colors is None or stream_colors.shape[0] != pts.shape[0]:
                colors.append([1.0, 1.0, 1.0, 0.9])
            else:
                colors.append(stream_colors[idx])
            if self._show_pathlines:
                self._update_pathline_item(idx_stream, pts[: idx + 1], stream_colors)
        if not positions:
            self._particle_item.setData(pos=np.empty((0, 3), dtype=np.float32))
            if self._show_pathlines:
                self._clear_pathlines()
            return
        self._particle_item.setData(pos=np.array(positions, dtype=np.float32), color=np.array(colors, dtype=np.float32))
        if self._show_pathlines:
            self._trim_pathlines(len(prepared_streamlines))

    def set_particle_step(self, step: int) -> None:
        self._particle_step = max(0, int(step))
        self._update_particles(self._particle_tracks)

    def _advance_particles(self) -> None:
        if not self._particles_enabled or not self._particle_tracks:
            return
        max_len = max((pts.shape[0] for pts, _colors in self._particle_tracks), default=0)
        if max_len <= 1:
            return
        next_step = self._particle_step + 1
        if next_step >= max_len:
            self._particle_cycle_index += 1
            if self._particle_cycle_index >= self._particle_cycles:
                self._particle_cycle_index = 0
            self._particle_step = 0
        else:
            self._particle_step = next_step
        self._update_particles(self._particle_tracks)

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

    def update_isosurface(self, volume, threshold: float, enabled: bool = True):
        self._iso_enabled = bool(enabled)
        self._iso_threshold = float(threshold)
        self._iso_volume = None if volume is None else np.asarray(volume, dtype=np.float32)
        self.clear_isosurface()
        if not self._iso_enabled or self._iso_volume is None:
            return
        if self._iso_volume.ndim != 3:
            return
        self._ensure_view()
        iso_data = np.transpose(self._iso_volume, (1, 0, 2))
        if not np.any(np.isfinite(iso_data)):
            return
        verts, faces = pg.isosurface(iso_data, level=self._iso_threshold)
        if verts.size == 0 or faces.size == 0:
            return
        mesh = gl.MeshData(vertexes=verts, faces=faces)
        self._iso_item = gl.GLMeshItem(
            meshdata=mesh,
            color=(0.2, 0.8, 1.0, 0.25),
            smooth=False,
            shader="shaded",
            drawEdges=False,
        )
        self._iso_item.setGLOptions("translucent")
        self.view.addItem(self._iso_item)

    def update_mask_isosurface(self, volume, enabled: bool = True):
        self._mask_iso_enabled = bool(enabled)
        self._mask_iso_volume = None if volume is None else np.asarray(volume, dtype=np.float32)
        self.clear_mask_isosurface()
        if not self._mask_iso_enabled or self._mask_iso_volume is None:
            return
        if self._mask_iso_volume.ndim != 3:
            return
        self._ensure_view()
        iso_data = np.transpose(self._mask_iso_volume, (1, 0, 2))
        if not np.any(np.isfinite(iso_data)):
            return
        verts, faces = pg.isosurface(iso_data, level=0.5)
        if verts.size == 0 or faces.size == 0:
            return
        mesh = gl.MeshData(vertexes=verts, faces=faces)
        self._mask_iso_item = gl.GLMeshItem(
            meshdata=mesh,
            color=(0.2, 1.0, 0.4, 0.18),
            smooth=False,
            shader="shaded",
            drawEdges=False,
        )
        self._mask_iso_item.setGLOptions("translucent")
        self.view.addItem(self._mask_iso_item)

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

    def _update_pathline_item(self, idx: int, pts: np.ndarray, colors: Optional[np.ndarray]) -> None:
        if not self._show_pathlines or pts.size == 0:
            return
        while len(self._pathline_items) <= idx:
            item = gl.GLLinePlotItem(width=1.0, antialias=True, mode="line_strip")
            self.view.addItem(item)
            self._pathline_items.append(item)
        item = self._pathline_items[idx]
        if colors is None or colors.shape[0] != pts.shape[0]:
            item.setData(pos=pts, color=(1.0, 1.0, 1.0, 0.9))
        else:
            item.setData(pos=pts, color=colors)

    def _trim_pathlines(self, target_len: int) -> None:
        if len(self._pathline_items) <= target_len:
            return
        for _ in range(len(self._pathline_items) - target_len):
            item = self._pathline_items.pop()
            try:
                self.view.removeItem(item)
            except Exception:
                pass

    def _clear_pathlines(self) -> None:
        for item in self._pathline_items:
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self._pathline_items = []


class StreamlineGalleryWindow(QtWidgets.QWidget):
    streamline_visibility_changed = QtCore.Signal(bool)

    def __init__(self, axis_orders, axis_flips, parent=None, columns: int = 4):
        super().__init__(parent)
        self.setWindowTitle("Streamline Gallery")
        self.views = []
        self._syncing_camera = False
        self._seed_phase = None
        self._seed_points = None
        self._show_streamlines = False

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
        self.show_streamlines_btn = QtWidgets.QPushButton("Show streamlines", self)
        self.show_streamlines_btn.setCheckable(True)
        self.show_streamlines_btn.clicked.connect(self._toggle_streamlines)
        control_row.addWidget(self.show_streamlines_btn)
        control_row.addStretch(1)
        layout.addLayout(control_row)
        layout.addWidget(scroll)

    def show_streamlines(self) -> bool:
        return self._show_streamlines

    def _toggle_streamlines(self, checked: bool) -> None:
        self._show_streamlines = bool(checked)
        label = "Hide streamlines" if self._show_streamlines else "Show streamlines"
        self.show_streamlines_btn.setText(label)
        self.streamline_visibility_changed.emit(self._show_streamlines)

    def clear_streamlines(self) -> None:
        for view, _order, _flips in self.views:
            view.update_streamlines([], None)
            view.update_contour(None, None)

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
    apply_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Streamline Player")
        self._phase = 1
        self._seed_points = None
        self._seed_phase = None
        self._precomputed_tracks = None
        self._precomputed_seed_phase = None
        self._precomputed_steps = None
        self._phase_timer = QtCore.QTimer(self)
        self._phase_timer.timeout.connect(self._advance_phase_playback)
        self._phase_cycle_index = 0
        self._phase_interval_ms = 100

        layout = QtWidgets.QVBoxLayout(self)
        top_row = QtWidgets.QHBoxLayout()

        controls_widget = QtWidgets.QWidget(self)
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        axis_row = QtWidgets.QHBoxLayout()
        axis_row.addWidget(QtWidgets.QLabel("Axis order"))
        self.axis_order_combo = QtWidgets.QComboBox(self)
        self.axis_order_combo.addItems(["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
        self.axis_order_combo.currentTextChanged.connect(lambda _val: self.phase_changed.emit(self._phase))
        axis_row.addWidget(self.axis_order_combo)
        axis_row.addWidget(QtWidgets.QLabel("Axis flips"))
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
        axis_row.addWidget(self.axis_flip_combo)
        controls_layout.addLayout(axis_row)

        seed_group = QtWidgets.QHBoxLayout()
        seed_group.addWidget(QtWidgets.QLabel("Seed count"))
        self.seed_spin = QtWidgets.QSpinBox(self)
        self.seed_spin.setRange(1, 5000)
        self.seed_spin.setValue(200)
        seed_group.addWidget(self.seed_spin)
        self.apply_btn = QtWidgets.QPushButton("Apply", self)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        seed_group.addWidget(self.apply_btn)
        pause_particles_btn = QtWidgets.QPushButton("Pause", self)
        pause_particles_btn.clicked.connect(self._on_pause_particles_clicked)
        seed_group.addWidget(pause_particles_btn)
        stop_particles_btn = QtWidgets.QPushButton("Stop", self)
        stop_particles_btn.clicked.connect(self._on_stop_particles_clicked)
        seed_group.addWidget(stop_particles_btn)
        controls_layout.addLayout(seed_group)

        timing_row = QtWidgets.QHBoxLayout()
        timing_row.addWidget(QtWidgets.QLabel("Seed phase"))
        self.seed_phase_spin = QtWidgets.QSpinBox(self)
        self.seed_phase_spin.setRange(1, 1)
        timing_row.addWidget(self.seed_phase_spin)
        timing_row.addWidget(QtWidgets.QLabel("Particle cycles"))
        self.particle_cycle_spin = QtWidgets.QSpinBox(self)
        self.particle_cycle_spin.setRange(1, 20)
        self.particle_cycle_spin.setValue(1)
        timing_row.addWidget(self.particle_cycle_spin)
        controls_layout.addLayout(timing_row)

        self.apply_mask_check = QtWidgets.QCheckBox("Apply to mask", self)
        self.apply_mask_check.setChecked(True)
        self.apply_mask_check.toggled.connect(lambda _val: self.phase_changed.emit(self._phase))
        controls_layout.addWidget(self.apply_mask_check)

        visibility_row = QtWidgets.QHBoxLayout()
        self.particle_check = QtWidgets.QCheckBox("Visualize particle", self)
        self.particle_check.setChecked(True)
        self.particle_check.toggled.connect(self._on_particle_toggle)
        visibility_row.addWidget(self.particle_check)
        self.pathline_check = QtWidgets.QCheckBox("Show pathline", self)
        self.pathline_check.setChecked(False)
        self.pathline_check.toggled.connect(self._on_pathline_toggle)
        visibility_row.addWidget(self.pathline_check)
        self.streamline_check = QtWidgets.QCheckBox("Show streamline", self)
        self.streamline_check.setChecked(False)
        self.streamline_check.toggled.connect(self._on_streamline_toggle)
        visibility_row.addWidget(self.streamline_check)
        controls_layout.addLayout(visibility_row)

        iso_row = QtWidgets.QHBoxLayout()
        self.iso_check = QtWidgets.QCheckBox("Show isosurface", self)
        self.iso_check.setChecked(False)
        self.iso_check.toggled.connect(lambda _val: self.phase_changed.emit(self._phase))
        iso_row.addWidget(self.iso_check)
        self.mask_iso_check = QtWidgets.QCheckBox("Show mask isosurface", self)
        self.mask_iso_check.setChecked(False)
        self.mask_iso_check.toggled.connect(lambda _val: self.phase_changed.emit(self._phase))
        iso_row.addWidget(self.mask_iso_check)
        iso_row.addWidget(QtWidgets.QLabel("Threshold"))
        self.iso_threshold_spin = QtWidgets.QDoubleSpinBox(self)
        self.iso_threshold_spin.setRange(0.0, 1e6)
        self.iso_threshold_spin.setDecimals(4)
        self.iso_threshold_spin.setSingleStep(0.1)
        self.iso_threshold_spin.setValue(0.1)
        self.iso_threshold_spin.valueChanged.connect(lambda _val: self.phase_changed.emit(self._phase))
        iso_row.addWidget(self.iso_threshold_spin)
        controls_layout.addLayout(iso_row)

        volume_row = QtWidgets.QHBoxLayout()
        self.iso_volume_label = QtWidgets.QLabel("Iso volume (mL): -")
        volume_row.addWidget(self.iso_volume_label)
        volume_row.addStretch(1)
        controls_layout.addLayout(volume_row)
        controls_layout.addStretch(1)

        top_row.addWidget(controls_widget, 1)
        self.view = StreamlineWindow(parent=self)
        self.view.set_particles_enabled(True, interval_ms=0)
        self.view.set_show_streamlines(False)
        self.view.set_show_pathlines(False)
        top_row.addWidget(self.view, 2)
        layout.addLayout(top_row)

        phase_row = QtWidgets.QHBoxLayout()
        self.phase_label = QtWidgets.QLabel("Phase: -")
        phase_row.addWidget(self.phase_label)
        self.phase_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.phase_slider.setRange(1, 1)
        self.phase_slider.setSingleStep(1)
        self.phase_slider.valueChanged.connect(self.set_phase)
        phase_row.addWidget(self.phase_slider)
        layout.addLayout(phase_row)

    def set_phase_count(self, count: int, seed_phase_max: Optional[int] = None) -> None:
        count = max(1, int(count))
        self.phase_slider.setRange(1, count)
        if seed_phase_max is None:
            seed_phase_max = count
        self.seed_phase_spin.setRange(1, max(1, int(seed_phase_max)))

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

    def seed_phase(self) -> int:
        return int(self.seed_phase_spin.value())

    def particle_cycles(self) -> int:
        return int(self.particle_cycle_spin.value())

    def apply_to_mask(self) -> bool:
        return self.apply_mask_check.isChecked()

    def set_seed_points(self, seed_points: np.ndarray, phase: int) -> None:
        self._seed_points = None if seed_points is None else np.asarray(seed_points, dtype=np.float32)
        self._seed_phase = int(phase)

    def seed_points(self) -> Optional[np.ndarray]:
        return self._seed_points

    def seed_phase_points(self) -> Optional[int]:
        return self._seed_phase

    def set_phase(self, phase: int) -> None:
        self._phase = int(phase)
        self.phase_label.setText(f"Phase: {self._phase}")
        if self.phase_slider.value() != self._phase:
            self.phase_slider.setValue(self._phase)
        self.phase_changed.emit(self._phase)

    def has_precomputed_tracks(self) -> bool:
        return self._precomputed_tracks is not None

    def set_precomputed_tracks(self, tracks, seed_phase: int, steps: int) -> None:
        self._precomputed_tracks = tracks
        self._precomputed_seed_phase = int(seed_phase)
        self._precomputed_steps = int(steps)

    def clear_precomputed_tracks(self) -> None:
        self._precomputed_tracks = None
        self._precomputed_seed_phase = None
        self._precomputed_steps = None

    def precomputed_seed_phase(self) -> Optional[int]:
        return self._precomputed_seed_phase

    def precomputed_steps(self) -> Optional[int]:
        return self._precomputed_steps

    def precomputed_tracks(self):
        return self._precomputed_tracks

    def isosurface_enabled(self) -> bool:
        return self.iso_check.isChecked()

    def isosurface_threshold(self) -> float:
        return float(self.iso_threshold_spin.value())

    def mask_isosurface_enabled(self) -> bool:
        return self.mask_iso_check.isChecked()

    def set_iso_volume(self, volume_ml: Optional[float]) -> None:
        if volume_ml is None or not np.isfinite(volume_ml):
            self.iso_volume_label.setText("Iso volume (mL): -")
            return
        self.iso_volume_label.setText(f"Iso volume (mL): {volume_ml:.2f}")

    def _on_apply_clicked(self) -> None:
        self.apply_requested.emit()

    def _on_stop_particles_clicked(self) -> None:
        self._phase_timer.stop()
        self.stop_requested.emit()

    def _on_pause_particles_clicked(self) -> None:
        self._phase_timer.stop()

    def _on_streamline_toggle(self, checked: bool) -> None:
        self.view.set_show_streamlines(bool(checked))
        self.phase_changed.emit(self._phase)

    def _on_pathline_toggle(self, checked: bool) -> None:
        self.view.set_show_pathlines(bool(checked))

    def _on_particle_toggle(self, checked: bool) -> None:
        if checked:
            self.view.set_particles_enabled(True, interval_ms=0)
        else:
            self.view.set_particles_enabled(False)

    def start_phase_playback(self) -> None:
        self._phase_cycle_index = 0
        if self._phase_interval_ms:
            self._phase_timer.setInterval(self._phase_interval_ms)
        self.view.reset_particle_animation()
        if self.phase_slider.maximum() > 1:
            self._phase_timer.start()

    def _advance_phase_playback(self) -> None:
        max_phase = int(self.phase_slider.maximum())
        if max_phase <= 1:
            return
        next_phase = self._phase + 1
        if next_phase > max_phase:
            next_phase = 1
        if not self.has_precomputed_tracks():
            seed_phase = int(self.seed_phase_spin.value())
            if next_phase == seed_phase:
                self._phase_cycle_index += 1
                if self._phase_cycle_index >= self.particle_cycles():
                    self._phase_cycle_index = 0
                    self.view.reset_particle_animation()
        self.set_phase(next_phase)


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
