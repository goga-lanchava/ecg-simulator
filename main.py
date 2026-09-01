"""ECG Simulator - entry point.

Wires the three pieces together and gets out of the way:

    SignalGenerator (QThread)  --writes-->  RingBuffer  <--reads--  MonitorView
         50 ms chunks                    10 s @ 1 kHz            QTimer, 30 FPS

The producer and the consumer never touch each other; the buffer's lock is the
only synchronisation point, so a slow repaint can never stall the sample clock.

    python main.py
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from core.buffer import RingBuffer
from core.generator import SignalGenerator
from core.state import SimulationState
from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    state = SimulationState()
    buffer = RingBuffer()
    generator = SignalGenerator(state, buffer)

    window = MainWindow(state.snapshot())
    window.bind(state, buffer, generator)

    generator.start()
    window.show()
    try:
        return app.exec()
    finally:
        generator.stop()          # belt and braces; closeEvent normally gets here first


if __name__ == "__main__":
    sys.exit(main())
