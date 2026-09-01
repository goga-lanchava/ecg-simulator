"""The right-hand monitor pane: two stacked traces on a black field.

The display does not scroll.  The x-axis is nailed to 0..10 s and the trace
stays where it was drawn; a block of ``NaN`` samples is blanked just ahead of
the buffer's write cursor, so what moves across the screen is the *gap*.  New
samples appear at its trailing edge and ten-second-old samples vanish at its
leading edge - the erase bar of a hospital monitor.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from core.buffer import RingBuffer
from core.state import BUFFER_SECONDS, SAMPLE_RATE

# Palette
BACKGROUND = "#000000"
AXIS = "#2a3441"
AXIS_TEXT = "#5b6b7d"
ECG_COLOR = "#39FF14"
RESP_COLOR = "#00FFFF"

# Fixed display windows.  A real monitor clips rather than rescaling: an
# autoranging trace would rubber-band on every artifact and make the rhythm
# impossible to read.
ECG_RANGE = (-1.2, 2.2)     # mV; a 1.8 mV PVC fits whole, a motion transient flat-tops
RESP_RANGE = (-1.5, 1.5)    # arbitrary units

SWEEP_FPS = 30
SWEEP_GAP_SAMPLES = 80      # erase bar width, 80 ms at 1 kHz

PEN_WIDTH = 2


def trace_pen(colour: str, width: int = PEN_WIDTH) -> pg.mkPen:
    """A width-2 pen that does not cost 60 ms a frame to draw.

    Qt strokes a normal wide pen by building an outline polygon around every
    segment - on a 10,000-point trace that alone blows the 33 ms frame budget,
    and the monitor visibly stalls once the buffer fills.  A *cosmetic* pen is
    stroked in device space by the rasteriser instead, which is ~15x cheaper
    and looks identical here because the trace is never scaled.
    """
    pen = pg.mkPen(colour, width=width)
    pen.setCosmetic(True)
    return pen


def apply_sweep_gap(data: np.ndarray, write_index: int, gap: int = SWEEP_GAP_SAMPLES):
    """Blank ``gap`` samples starting at the write cursor, wrapping at the end.

    Operates on the last axis, so it takes the whole ``(channels, n)`` block at
    once and keeps every channel's gap in the same place.  Mutates in place -
    callers pass the copy that :meth:`RingBuffer.snapshot` already handed them.
    """
    n = data.shape[-1]
    gap = min(int(gap), n)
    end = write_index + gap
    if end <= n:
        data[..., write_index:end] = np.nan
    else:                                   # the bar is straddling the wrap point
        data[..., write_index:] = np.nan
        data[..., : end - n] = np.nan
    return data


class MonitorView(QWidget):
    """Stacked ECG (top) and respiration (bottom) plots."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        duration: float = BUFFER_SECONDS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.n_samples = int(self.sample_rate * self.duration)

        # One shared x-axis for both traces; only the y data ever changes, which
        # is what makes the sweep cheap to redraw.
        self._x = np.arange(self.n_samples, dtype=np.float64) / self.sample_rate

        # Antialiasing is not what costs us frames here (measured: no effect),
        # but on a 10 k-point trace it buys nothing visible either.
        pg.setConfigOptions(antialias=False)

        self.ecg_plot = self._make_plot("ECG", "mV", ECG_RANGE, show_time_axis=False)
        self.resp_plot = self._make_plot("RESP", "a.u.", RESP_RANGE, show_time_axis=True)

        self.ecg_curve = self.ecg_plot.plot(
            self._x, np.full(self.n_samples, np.nan), pen=trace_pen(ECG_COLOR)
        )
        self.resp_curve = self.resp_plot.plot(
            self._x, np.full(self.n_samples, np.nan), pen=trace_pen(RESP_COLOR)
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.ecg_plot, stretch=2)     # ECG gets the room; detail matters
        layout.addWidget(self.resp_plot, stretch=1)

        # Consumer side of the producer/consumer split: the timer samples
        # whatever the generator thread has written, at its own independent rate.
        self._buffer: RingBuffer | None = None
        self._ecg_index = 0
        self._resp_index = 1
        self.gap_samples = SWEEP_GAP_SAMPLES

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.refresh)

    def _make_plot(self, title: str, units: str, y_range, show_time_axis: bool):
        widget = pg.PlotWidget()
        widget.setBackground(BACKGROUND)

        item = widget.getPlotItem()
        item.showGrid(x=False, y=False)
        item.setMouseEnabled(x=False, y=False)      # a monitor is not pannable
        item.setMenuEnabled(False)
        item.hideButtons()
        item.disableAutoRange()                     # fixed window; clip, do not rescale
        item.setXRange(0.0, self.duration, padding=0.0)
        item.setYRange(*y_range, padding=0.0)

        item.setLabel("left", title, units=units, color=AXIS_TEXT, size="9pt")
        for edge in ("left", "bottom"):
            axis = item.getAxis(edge)
            axis.setPen(AXIS)
            axis.setTextPen(AXIS_TEXT)
        if show_time_axis:
            item.setLabel("bottom", "seconds", color=AXIS_TEXT, size="9pt")
        else:
            item.getAxis("bottom").setStyle(showValues=False)
        return widget

    # -- sweep ----------------------------------------------------------------
    def attach(self, buffer: RingBuffer) -> None:
        """Point the monitor at the buffer the producer is filling."""
        if buffer.capacity != self.n_samples:
            raise ValueError(
                f"buffer holds {buffer.capacity} samples, display expects {self.n_samples}"
            )
        self._buffer = buffer
        self._ecg_index = buffer.channel_index("ecg")
        self._resp_index = buffer.channel_index("resp")

    def start(self, fps: int = SWEEP_FPS) -> None:
        if self._buffer is None:
            raise RuntimeError("attach() a buffer before starting the sweep")
        self._timer.start(max(1, round(1000 / fps)))

    def stop(self) -> None:
        self._timer.stop()

    @property
    def is_running(self) -> bool:
        return self._timer.isActive()

    def refresh(self) -> None:
        """One frame: read the buffer, cut the erase bar, redraw."""
        if self._buffer is None:
            return
        data, write_index = self._buffer.snapshot()     # already a private copy
        apply_sweep_gap(data, write_index, self.gap_samples)
        self.update_traces(data[self._ecg_index], data[self._resp_index])

    # -- rendering ------------------------------------------------------------
    def update_traces(self, ecg: np.ndarray, resp: np.ndarray) -> None:
        """Redraw both traces.  ``NaN`` samples render as gaps, not zeros."""
        self.ecg_curve.setData(self._x, ecg)
        self.resp_curve.setData(self._x, resp)

    def clear(self) -> None:
        blank = np.full(self.n_samples, np.nan)
        self.update_traces(blank, blank)
