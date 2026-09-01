"""Generate the application icon: a green QRS complex on black.

Drawn from the app's own palette rather than shipped as a binary blob, so it
stays in step if the colours change.

    python packaging/make_icon.py

Writes packaging/icon.ico (Windows) and packaging/icon.png (source for macOS,
which needs a .icns produced by iconutil on a Mac).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QPointF, Qt                      # noqa: E402
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen  # noqa: E402
from PyQt6.QtWidgets import QApplication                  # noqa: E402

from ui.monitor import BACKGROUND, ECG_COLOR              # noqa: E402

SIZES = (256, 128, 64, 48, 32, 16)

# One beat in normalised (x, y) space, y up.  Deliberately exaggerated: a real
# P wave is invisible at 16 px.
BEAT = [
    (0.00, 0.00), (0.10, 0.00), (0.16, 0.10), (0.22, 0.00), (0.32, 0.00),
    (0.38, -0.16), (0.46, 0.92), (0.54, -0.34), (0.60, 0.00), (0.70, 0.00),
    (0.80, 0.26), (0.90, 0.00), (1.00, 0.00),
]


def render(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(QColor(BACKGROUND))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, size >= 32)

    margin = size * 0.10
    width = size - 2 * margin
    # Centred on the drawing, not the axis: the R wave is far taller than the
    # S is deep, so an axis at mid-height leaves the glyph sitting high.
    midline = size * 0.60
    scale = size * 0.34

    path = QPainterPath()
    for index, (x, y) in enumerate(BEAT):
        point = QPointF(margin + x * width, midline - y * scale)
        path.moveTo(point) if index == 0 else path.lineTo(point)

    pen = QPen(QColor(ECG_COLOR))
    pen.setWidthF(max(1.4, size * 0.075))
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawPath(path)
    painter.end()
    return image


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])   # noqa: F841
    here = os.path.dirname(os.path.abspath(__file__))

    largest = render(SIZES[0])
    png = os.path.join(here, "icon.png")
    largest.save(png, "PNG")
    print(f"wrote {png}")

    # Qt writes a single-resolution .ico; Windows scales it acceptably, and the
    # 256 px source keeps the taskbar icon crisp.
    ico = os.path.join(here, "icon.ico")
    if not largest.save(ico, "ICO"):
        print("ICO writer unavailable on this Qt build; .exe will use the default icon")
        return 1
    print(f"wrote {ico}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
