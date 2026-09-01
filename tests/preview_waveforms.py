"""Render the generator's output to a PNG for eyeball verification.

Not a test - a dev tool.  Runs headless via the offscreen Qt platform and uses
the same dark palette and pens the monitor will use in Steps 3 and 4, so it
doubles as an early check on the rendering stack.

    python tests/preview_waveforms.py [out.png]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyqtgraph as pg                                   # noqa: E402
import pyqtgraph.exporters                               # noqa: E402,F401
from PyQt6.QtCore import QRectF                          # noqa: E402
from PyQt6.QtGui import QFont                            # noqa: E402
from PyQt6.QtWidgets import QApplication                 # noqa: E402

from core.generator import WaveformEngine                # noqa: E402
from core.state import SAMPLE_RATE, SimulationState      # noqa: E402

ECG_PEN = pg.mkPen("#39FF14", width=2)
RESP_PEN = pg.mkPen("#00FFFF", width=2)


def render(engine: WaveformEngine, seconds: float, chunk: int = 50):
    ecg, resp = [], []
    for _ in range(int(seconds * SAMPLE_RATE / chunk)):
        block = engine.next_chunk(chunk)
        ecg.append(block["ecg"])
        resp.append(block["resp"])
    return np.concatenate(ecg), np.concatenate(resp)


def build(**params) -> WaveformEngine:
    return WaveformEngine(SimulationState(**params), SAMPLE_RATE, np.random.default_rng(11))


def panels():
    # 1. Clean sinus rhythm.
    e = build(heart_rate=72)
    ecg, _ = render(e, 6.0)
    yield "Normal sinus rhythm, 72 bpm - clean", ecg, ECG_PEN

    # 2. Tachycardia: Bazett scaling pulls P and T in toward the QRS.
    e = build(heart_rate=180)
    ecg, _ = render(e, 6.0)
    yield "Sinus tachycardia, 180 bpm", ecg, ECG_PEN

    # 3. One PVC with its compensatory pause.
    e = build(heart_rate=72)
    pre, _ = render(e, 2.5)          # show the rhythm the ectopic beat interrupts
    e.state.trigger_pvc()
    post, _ = render(e, 5.5)
    yield ("PVC: wide ectopic beat, no P wave, discordant T, full compensatory pause",
           np.concatenate([pre, post]), ECG_PEN)

    # 4. The full noise stack.
    e = build(heart_rate=72, mains_amplitude=0.12, baseline_amplitude=0.25, gaussian_sigma=0.03)
    ecg, _ = render(e, 6.0)
    yield "Noise: 50 Hz mains + 0.5 Hz baseline wander + Gaussian", ecg, ECG_PEN

    # 5. Motion artifact, then recovery.
    e = build(heart_rate=72)
    pre, _ = render(e, 2.0)
    e.state.trigger_motion()
    post, _ = render(e, 6.0)
    yield ("Motion artifact injected at 2.0 s: transient, then clean recovery",
           np.concatenate([pre, post]), ECG_PEN)

    # 6. Respiration.
    e = build(heart_rate=72, respiratory_rate=14)
    _, resp = render(e, 20.0)
    yield "Respiration, 14 brpm - skewed, I:E 1:1.5", resp, RESP_PEN


WIDTH, HEIGHT = 1500, 1180


def main(out: str) -> None:
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Segoe UI", 9))
    pg.setConfigOptions(antialias=True)

    layout = pg.GraphicsLayoutWidget(size=(WIDTH, HEIGHT), show=False)
    layout.setBackground("#000000")

    plots = []
    for row, (title, y, pen) in enumerate(panels()):
        plot = layout.addPlot(row=row, col=0, title=title)
        plots.append(plot)
        plot.titleLabel.setAttr("color", "#8899aa")
        plot.titleLabel.setAttr("size", "10pt")
        plot.showGrid(x=False, y=False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.getAxis("left").setPen("#334455")
        plot.getAxis("bottom").setPen("#334455")
        plot.getAxis("left").setTextPen("#667788")
        plot.getAxis("bottom").setTextPen("#667788")
        plot.plot(np.arange(y.size) / SAMPLE_RATE, y, pen=pen)
        plot.setLabel("bottom", "seconds")

    # A widget that is never shown gets no resize event, so the scene rect keeps
    # its tiny default and every plot collapses to a flat line.  Drive the layout
    # explicitly instead of relying on the window manager.
    rect = QRectF(0, 0, WIDTH, HEIGHT)
    layout.ci.setGeometry(rect)
    layout.scene().setSceneRect(rect)
    for plot in plots:
        plot.autoRange()
    app.processEvents()

    exporter = pg.exporters.ImageExporter(layout.scene())
    exporter.parameters()["width"] = WIDTH
    exporter.export(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "waveform_preview.png")
