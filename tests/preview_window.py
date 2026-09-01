"""Screenshot the running app mid-sweep, for layout and rendering review.

Dev tool, not a test.  Runs the real producer thread for one full lap plus a
little, so the erase bar sits partway across a fully populated buffer, and
fires both injections on the way so the frame has something to show.

    python tests/preview_window.py [out.png] [rhythm] [breathing]
    python tests/preview_window.py afib.png AFib Cheyne-Stokes
"""

from __future__ import annotations

import os
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.buffer import RingBuffer                       # noqa: E402
from core.generator import SignalGenerator               # noqa: E402
from core.state import SimulationState                   # noqa: E402
from ui.main_window import MainWindow                     # noqa: E402

# One 10 s lap, then 2.5 s more so the cursor sits a quarter of the way across.
CAPTURE_AT_MS = 12_500


def main(out: str, rhythm: str = "Sinus", breathing: str = "Normal") -> int:
    app = QApplication(sys.argv[:1])

    state = SimulationState(heart_rate=72, respiratory_rate=15,
                            mains_amplitude=0.05, gaussian_sigma=0.010,
                            cardiac_rhythm=rhythm, resp_pattern=breathing)
    buffer = RingBuffer()
    generator = SignalGenerator(state, buffer)

    window = MainWindow(state.snapshot())
    window.bind(state, buffer, generator)
    window.resize(1400, 820)
    window.show()
    generator.start()

    # Land both events inside the visible 10 s window at capture time.
    QTimer.singleShot(CAPTURE_AT_MS - 8200, state.trigger_pvc)
    QTimer.singleShot(CAPTURE_AT_MS - 4200, state.trigger_motion)

    def shoot() -> None:
        window.grab().save(out)
        print(f"wrote {out}")
        window.close()
        app.quit()

    QTimer.singleShot(CAPTURE_AT_MS, shoot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:] or ["window_preview.png"])))
