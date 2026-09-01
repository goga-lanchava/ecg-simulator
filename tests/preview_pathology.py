"""Render the whole rhythm and breathing catalogue to PNGs for eyeball review.

Dev tool, not a test.  Same palette and pens as the monitor.

    python tests/preview_pathology.py [prefix]

Writes <prefix>_rhythms.png and <prefix>_breathing.png.
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
from core.pathology import RESP_SPECS, RHYTHMS           # noqa: E402
from core.state import SAMPLE_RATE, SimulationState      # noqa: E402

ECG_PEN = pg.mkPen("#39FF14", width=2)
RESP_PEN = pg.mkPen("#00FFFF", width=2)

RHYTHM_SECONDS = 8.0
RESP_SECONDS = 150.0


def render(engine: WaveformEngine, seconds: float, chunk: int = 50):
    ecg, resp = [], []
    for _ in range(int(seconds * SAMPLE_RATE / chunk)):
        block = engine.next_chunk(chunk)
        ecg.append(block["ecg"])
        resp.append(block["resp"])
    return np.concatenate(ecg), np.concatenate(resp)


def build(**params) -> WaveformEngine:
    return WaveformEngine(SimulationState(**params), SAMPLE_RATE, np.random.default_rng(11))


def sheet(panels, columns: int, width: int, row_height: int, out: str) -> None:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setFont(QFont("Segoe UI", 9))
    pg.setConfigOptions(antialias=True)      # one-shot export: quality over speed

    rows = (len(panels) + columns - 1) // columns
    height = rows * row_height
    layout = pg.GraphicsLayoutWidget(size=(width, height), show=False)
    layout.setBackground("#000000")

    plots = []
    for index, (title, y, pen) in enumerate(panels):
        plot = layout.addPlot(row=index // columns, col=index % columns, title=title)
        plots.append(plot)
        plot.titleLabel.setAttr("color", "#8899aa")
        plot.titleLabel.setAttr("size", "9pt")
        plot.showGrid(x=False, y=False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        for edge in ("left", "bottom"):
            plot.getAxis(edge).setPen("#334455")
            plot.getAxis(edge).setTextPen("#667788")
        plot.plot(np.arange(y.size) / SAMPLE_RATE, y, pen=pen)
        plot.setLabel("bottom", "seconds")

    rect = QRectF(0, 0, width, height)
    layout.ci.setGeometry(rect)
    layout.scene().setSceneRect(rect)
    for plot in plots:
        plot.autoRange()
    app.processEvents()

    exporter = pg.exporters.ImageExporter(layout.scene())
    exporter.parameters()["width"] = width
    exporter.export(out)
    print(f"wrote {out}  ({len(panels)} panels)")


def main(prefix: str = "catalogue") -> None:
    rhythms = []
    for name, spec in RHYTHMS.items():
        ecg, _ = render(build(heart_rate=72, cardiac_rhythm=name), RHYTHM_SECONDS)
        rhythms.append((f"{name} - {spec.note}", ecg, ECG_PEN))
    sheet(rhythms, columns=2, width=1900, row_height=185, out=f"{prefix}_rhythms.png")

    breathing = []
    for name, spec in RESP_SPECS.items():
        _, resp = render(build(respiratory_rate=15, resp_pattern=name), RESP_SECONDS)
        breathing.append((f"{name} - {spec.note}", resp, RESP_PEN))
    sheet(breathing, columns=1, width=1500, row_height=180, out=f"{prefix}_breathing.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "catalogue")
