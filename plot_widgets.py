import numpy as np
from PySide6 import QtCore, QtWidgets

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
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def plot_metric(self, phases, values, label: str, color: str):
        self.ax.clear()
        self.ax.grid(True)
        self._phases = phases
        self._values = values
        self.ax.plot(phases, values, marker="o", color=color, label=label)
        self.ax.set_xlabel("Phase")
        self.ax.set_ylabel(label)
        self.ax.legend(loc="upper right")
        try:
            self.fig.tight_layout()
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
