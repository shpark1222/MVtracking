import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph.opengl as gl

import matplotlib

matplotlib.use("QtAgg")
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


class StreamlineWindow(QtWidgets.QWidget):
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
        self._volume_shape = None

        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = 200
        self.view.setBackgroundColor("k")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.view)

    def clear_streamlines(self):
        for item in self._line_items:
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self._line_items = []

    def update_streamlines(self, streamlines, volume_shape):
        self.clear_streamlines()
        self._volume_shape = volume_shape
        if not streamlines:
            return
        self._update_view_center(volume_shape)
        for line in streamlines:
            if line is None or len(line) == 0:
                continue
            pts = self._transform_points(np.asarray(line, dtype=np.float32), volume_shape)
            item = gl.GLLinePlotItem(
                pos=pts,
                color=(0.2, 0.8, 1.0, 0.9),
                width=1.0,
                antialias=True,
                mode="line_strip",
            )
            self.view.addItem(item)
            self._line_items.append(item)

    def _update_view_center(self, volume_shape):
        if volume_shape is None:
            return
        shape_xyz = np.array([volume_shape[1], volume_shape[0], volume_shape[2]], dtype=float)
        center = shape_xyz / 2.0
        self.view.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))

    def _transform_points(self, points: np.ndarray, volume_shape):
        if points.size == 0:
            return points
        xyz = points[:, [1, 0, 2]].astype(np.float32)
        shape_xyz = np.array([volume_shape[1], volume_shape[0], volume_shape[2]], dtype=np.float32)
        flips = self.axis_flips if self.axis_flips is not None else (False, False, False)
        for axis, flip in enumerate(flips[:3]):
            if flip:
                xyz[:, axis] = (shape_xyz[axis] - 1.0) - xyz[:, axis]
        axis_map = {"X": 0, "Y": 1, "Z": 2}
        perm = [axis_map[c] for c in self.axis_order]
        return xyz[:, perm]
