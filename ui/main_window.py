"""Main window: control panel on the left, monitor on the right, 1:3.

The panel emits intent keyed by *state field name* rather than by widget, so
:meth:`MainWindow.bind` connects the whole thing in one hop.  It owns no
simulation state of its own, which is what lets it be built and tested without
a running generator.

The panel collapses (Ctrl+H, F9, or the rail button) to give the monitor the
full width; the toggle deliberately lives outside the panel so it survives the
panel being hidden.
"""

from __future__ import annotations

import os
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.pathology import (
    RESP_SPECS,
    RHYTHMS,
    SHOCK_KINDS,
    THERAPY_CARDIOVERT,
    THERAPY_DEFIB,
    URGENCY_LETHAL,
    therapy_for,
)
from core.state import (
    BASELINE_RANGE,
    CARDIAC_RHYTHMS,
    RESP_PATTERNS,
    GAUSSIAN_RANGE,
    HR_RANGE,
    MAINS_RANGE,
    RR_RANGE,
    StateSnapshot,
)
from ui.monitor import ECG_COLOR, RESP_COLOR, MonitorView

COLLAPSE_GLYPH = "‹"     # single left angle quote
EXPAND_GLYPH = "›"       # single right angle quote

ALARM_FLASH_MS = 550     # slow enough to read, fast enough to read as an alarm

SHOCK_MESSAGES = {
    "converted": "Shock delivered - rhythm converted to sinus.",
    "persists": "Shock delivered - rhythm unchanged. Resume CPR and recharge.",
    "no_effect": "Shock delivered - no effect. This rhythm is not shockable.",
    "induced_vf": "Shock landed on the T wave - VENTRICULAR FIBRILLATION induced.",
    "not_delivered": "Not delivered - no R wave to synchronise to. Use unsynchronised.",
}

STYLESHEET = f"""
QWidget#root {{ background: #05070a; }}
QWidget#panel {{ background: #0d1219; }}

QLabel#appTitle {{
    color: #e6edf3; font-size: 13px; font-weight: 600; letter-spacing: 1px;
}}
QLabel#appSubtitle {{ color: #4d5d70; font-size: 10px; letter-spacing: 1px; }}
QLabel#section {{
    color: #7d8ea3; font-size: 10px; font-weight: 600; letter-spacing: 2px;
    padding-top: 4px;
}}
QLabel#controlName {{ color: #b6c2cf; font-size: 11px; }}
QLabel#controlValue {{ color: {ECG_COLOR}; font-size: 11px; font-weight: 600; }}

QFrame#divider {{ background: #1b2330; max-height: 1px; border: none; }}

QSlider::groove:horizontal {{
    height: 4px; background: #1b2330; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: #2f6f4f; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #c9d6e2; width: 12px; height: 12px;
    margin: -5px 0; border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: {ECG_COLOR}; }}

QPushButton {{
    background: #161e29; color: #c9d6e2; border: 1px solid #2a3646;
    border-radius: 4px; padding: 8px; font-size: 11px; text-align: center;
}}
QPushButton:hover {{ background: #1d2734; border-color: #3b4b60; }}
QPushButton:pressed {{ background: #263243; }}
QPushButton#motion {{ border-color: #6b4a1f; color: #e0a44c; }}
QPushButton#motion:hover {{ background: #2a1f10; }}
QPushButton#pvc {{ border-color: #6b2530; color: #e06c78; }}
QPushButton#pvc:hover {{ background: #2a1216; }}

QGroupBox {{
    color: #7d8ea3; font-size: 10px; font-weight: 600; letter-spacing: 2px;
    border: 1px solid #1b2330; border-radius: 4px;
    margin-top: 10px; padding: 10px 8px 6px 8px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QGroupBox QLabel {{
    color: #b6c2cf; font-size: 11px; font-weight: 400; letter-spacing: 0;
}}

QComboBox {{
    background: #161e29; color: #c9d6e2; border: 1px solid #2a3646;
    border-radius: 3px; padding: 4px 6px; font-size: 11px;
}}
QComboBox:hover {{ border-color: #3b4b60; }}
QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{
    background: #0d1219; color: #c9d6e2; border: 1px solid #2a3646;
    selection-background-color: #2f6f4f; outline: none;
}}

QLabel#summary {{ color: #8494a6; font-size: 10px; }}

QScrollArea#panel {{ background: #0d1219; border: none; }}
QScrollArea#panel > QWidget > QWidget {{ background: #0d1219; }}
QScrollBar:vertical {{
    background: #0d1219; width: 8px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #2a3646; border-radius: 4px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: #3b4b60; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QPushButton#defib {{ border-color: #6b2530; color: #ff6b7a; font-weight: 600; }}
QPushButton#defib:hover {{ background: #2a1216; }}
QPushButton#cardiovert {{ border-color: #4a5a2a; color: #b6d46a; font-weight: 600; }}
QPushButton#cardiovert:hover {{ background: #1c2413; }}

QCheckBox {{ color: #b6c2cf; font-size: 11px; spacing: 6px; padding-top: 4px; }}
QCheckBox::indicator {{
    width: 12px; height: 12px; border: 1px solid #2a3646;
    border-radius: 2px; background: #161e29;
}}
QCheckBox::indicator:checked {{ background: #2f6f4f; border-color: {ECG_COLOR}; }}

QLabel#alarm {{
    font-size: 12px; font-weight: 600; letter-spacing: 1px; padding: 7px 12px;
}}

QPushButton#rail {{
    background: #0d1219; color: #8494a6; border: none; border-radius: 0;
    border-right: 1px solid #1b2330; padding: 0; font-size: 14px;
}}
QPushButton#rail:hover {{ background: #1d2734; color: {ECG_COLOR}; }}
"""


class LabeledSlider(QWidget):
    """A slider that reports a float in engineering units, with a live readout.

    Qt sliders are integer-only, so each control carries its own mapping from
    step index to physical value rather than scattering conversions at the
    call sites.
    """

    changed = pyqtSignal(str, float)

    def __init__(
        self,
        key: str,
        label: str,
        limits: tuple[float, float],
        value: float,
        steps: int,
        unit: str = "",
        decimals: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = key
        self.low, self.high = limits
        self.steps = int(steps)
        self.unit = unit
        self.decimals = int(decimals)

        self.name_label = QLabel(label, objectName="controlName")
        self.value_label = QLabel(objectName="controlValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.steps)
        self.slider.setValue(self._to_step(value))
        self.slider.valueChanged.connect(self._on_slider)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.name_label)
        header.addStretch(1)
        header.addWidget(self.value_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self.slider)

        self._refresh_label(value)

    # -- value mapping --------------------------------------------------------
    def _to_step(self, value: float) -> int:
        span = self.high - self.low
        frac = 0.0 if span == 0 else (value - self.low) / span
        return int(round(min(1.0, max(0.0, frac)) * self.steps))

    def _to_value(self, step: int) -> float:
        return self.low + (self.high - self.low) * step / self.steps

    def value(self) -> float:
        return self._to_value(self.slider.value())

    def set_value(self, value: float) -> None:
        """Move the slider without re-emitting - used to sync from state."""
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(self._to_step(value))
        self.slider.blockSignals(blocked)
        self._refresh_label(self.value())

    def _refresh_label(self, value: float) -> None:
        text = f"{value:.{self.decimals}f}"
        self.value_label.setText(f"{text} {self.unit}".strip())

    def _on_slider(self, step: int) -> None:
        value = self._to_value(step)
        self._refresh_label(value)
        self.changed.emit(self.key, value)


class ControlPanel(QWidget):
    """Left pane.  Emits intent; owns no simulation state."""

    parameter_changed = pyqtSignal(str, float)     # (state field name, value)
    motion_requested = pyqtSignal()
    pvc_requested = pyqtSignal()
    export_requested = pyqtSignal()
    shock_requested = pyqtSignal(str)              # 'defibrillate' | 'cardiovert'
    alarm_toggled = pyqtSignal(bool)

    def __init__(self, defaults: StateSnapshot | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        defaults = defaults or StateSnapshot()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        title = QLabel("ECG SIMULATOR", objectName="appTitle")
        subtitle = QLabel("SYNTHETIC PATIENT MONITOR", objectName="appSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(6)

        self.sliders: dict[str, LabeledSlider] = {}

        layout.addWidget(self._section("RHYTHM"))
        layout.addWidget(self._divider())
        self._add_slider(layout, "heart_rate", "Heart Rate", HR_RANGE,
                         defaults.heart_rate, steps=170, unit="bpm")
        self._add_slider(layout, "respiratory_rate", "Respiratory Rate", RR_RANGE,
                         defaults.respiratory_rate, steps=22, unit="brpm")

        layout.addSpacing(8)
        layout.addWidget(self._section("NOISE"))
        layout.addWidget(self._divider())
        self._add_slider(layout, "mains_amplitude", "Mains 50 Hz", MAINS_RANGE,
                         defaults.mains_amplitude, steps=100, unit="mV", decimals=2)
        self._add_slider(layout, "baseline_amplitude", "Baseline Wander", BASELINE_RANGE,
                         defaults.baseline_amplitude, steps=100, unit="mV", decimals=2)
        self._add_slider(layout, "gaussian_sigma", "Gaussian Noise", GAUSSIAN_RANGE,
                         defaults.gaussian_sigma, steps=100, unit="mV", decimals=3)

        layout.addSpacing(8)
        disease_group = QGroupBox("PATHOLOGY OVERRIDES")
        disease_layout = QFormLayout()
        disease_layout.setContentsMargins(4, 4, 4, 0)
        disease_layout.setSpacing(6)

        self.rhythm_combo = QComboBox()
        self.rhythm_combo.addItems(CARDIAC_RHYTHMS)
        self.rhythm_combo.setCurrentText(defaults.cardiac_rhythm)

        self.resp_combo = QComboBox()
        self.resp_combo.addItems(RESP_PATTERNS)
        self.resp_combo.setCurrentText(defaults.resp_pattern)

        disease_layout.addRow("ECG Rhythm:", self.rhythm_combo)
        disease_layout.addRow("Breathing:", self.resp_combo)
        disease_group.setLayout(disease_layout)
        layout.addWidget(disease_group)

        layout.addSpacing(8)
        layout.addWidget(self._section("EVENTS"))
        layout.addWidget(self._divider())
        self.motion_button = QPushButton("Inject Motion Artifact", objectName="motion")
        self.motion_button.clicked.connect(self.motion_requested)
        layout.addWidget(self.motion_button)

        self.pvc_button = QPushButton("Inject PVC", objectName="pvc")
        self.pvc_button.setToolTip("Premature Ventricular Contraction")
        self.pvc_button.clicked.connect(self.pvc_requested)
        layout.addWidget(self.pvc_button)

        layout.addSpacing(8)
        layout.addWidget(self._section("THERAPY"))
        layout.addWidget(self._divider())
        self.defib_button = QPushButton("Defibrillate  (unsync)", objectName="defib")
        self.defib_button.setToolTip(
            "Unsynchronised shock - for pulseless VF/VT. "
            "On an organised rhythm this can land on the T wave and cause VF.")
        self.defib_button.clicked.connect(
            lambda: self.shock_requested.emit(THERAPY_DEFIB))
        layout.addWidget(self.defib_button)

        self.cardiovert_button = QPushButton("Cardiovert  (sync)", objectName="cardiovert")
        self.cardiovert_button.setToolTip(
            "Synchronised shock - timed to the R wave. "
            "Cannot be delivered without an organised QRS.")
        self.cardiovert_button.clicked.connect(
            lambda: self.shock_requested.emit(THERAPY_CARDIOVERT))
        layout.addWidget(self.cardiovert_button)

        self.alarm_check = QCheckBox("Therapy alarm")
        self.alarm_check.setToolTip(
            "Alert when the current rhythm calls for electrical therapy.")
        self.alarm_check.setChecked(defaults.alarm_enabled)
        self.alarm_check.toggled.connect(self.alarm_toggled)
        layout.addWidget(self.alarm_check)

        layout.addSpacing(8)
        layout.addWidget(self._section("DATA"))
        layout.addWidget(self._divider())
        self.export_button = QPushButton("Export Buffer to CSV")
        self.export_button.clicked.connect(self.export_requested)
        layout.addWidget(self.export_button)

        # What the current selection actually changes.  Reads the catalogue, so
        # a rhythm added to core.pathology documents itself here for free.
        layout.addSpacing(10)
        layout.addWidget(self._divider())
        self.summary_label = QLabel(objectName="summary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.summary_label)
        for combo in (self.rhythm_combo, self.resp_combo):
            combo.currentTextChanged.connect(self._refresh_summary)
        self._refresh_summary()

        layout.addStretch(1)

        self.status_label = QLabel("", objectName="appSubtitle")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # -- selection summary ----------------------------------------------------
    def describe_selection(self) -> str:
        """Plain-text account of what the two dropdowns currently change."""
        lines = []
        rhythm = self.rhythm_combo.currentText()
        spec = RHYTHMS.get(rhythm)
        if spec is not None:
            lines.append(f"ECG - {rhythm}")
            lines.append(spec.note)
            if spec.rate_bpm is not None:
                lines.append(f"Rate fixed at {spec.rate_bpm:.0f} bpm - "
                             "the Heart Rate slider has no effect.")
            elif spec.scheduler in ("wenckebach", "mobitz2", "complete_block"):
                lines.append("Heart Rate sets the atrial rate; some beats are "
                             "not conducted, so the pulse is slower.")
            else:
                lines.append("Rate follows the Heart Rate slider.")

        breathing = self.resp_combo.currentText()
        rspec = RESP_SPECS.get(breathing)
        if rspec is not None:
            lines.append("")
            lines.append(f"BREATHING - {breathing}")
            lines.append(rspec.note)
            if rspec.rate_bpm is not None:
                lines.append(f"Rate fixed at {rspec.rate_bpm:.0f} brpm - "
                             "the Respiratory Rate slider has no effect.")
            else:
                lines.append("Rate follows the Respiratory Rate slider.")
        return "\n".join(lines)

    def _refresh_summary(self, *_args) -> None:
        rhythm = self.rhythm_combo.currentText()
        breathing = self.resp_combo.currentText()
        spec, rspec = RHYTHMS.get(rhythm), RESP_SPECS.get(breathing)

        blocks = []
        if spec is not None:
            detail = [spec.note]
            if spec.rate_bpm is not None:
                detail.append(f"Rate fixed at <b>{spec.rate_bpm:.0f} bpm</b> &mdash; "
                              "the Heart Rate slider has no effect.")
            elif spec.scheduler in ("wenckebach", "mobitz2", "complete_block"):
                detail.append("Heart Rate sets the <i>atrial</i> rate; dropped "
                              "beats make the pulse slower.")
            else:
                detail.append("Rate follows the Heart Rate slider.")
            blocks.append(
                f"<span style='color:{ECG_COLOR}'><b>ECG &middot; {rhythm}</b></span>"
                f"<br>{'<br>'.join(detail)}")

        if rspec is not None:
            detail = [rspec.note]
            if rspec.rate_bpm is not None:
                detail.append(f"Rate fixed at <b>{rspec.rate_bpm:.0f} brpm</b> &mdash; "
                              "the Respiratory Rate slider has no effect.")
            blocks.append(
                f"<span style='color:{RESP_COLOR}'><b>BREATHING &middot; {breathing}"
                f"</b></span><br>{'<br>'.join(detail)}")

        self.summary_label.setText("<br><br>".join(blocks))

    # -- construction helpers -------------------------------------------------
    @staticmethod
    def _section(text: str) -> QLabel:
        return QLabel(text, objectName="section")

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame(objectName="divider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _add_slider(self, layout: QVBoxLayout, key: str, label: str,
                    limits, value: float, steps: int, unit: str = "",
                    decimals: int = 0) -> None:
        control = LabeledSlider(key, label, limits, value, steps, unit, decimals)
        control.changed.connect(self.parameter_changed)
        self.sliders[key] = control
        layout.addWidget(control)

    # -- feedback -------------------------------------------------------------
    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def sync(self, params: StateSnapshot) -> None:
        """Push state back into the widgets without re-emitting change signals."""
        for key, control in self.sliders.items():
            control.set_value(getattr(params, key))
        for combo, value in ((self.rhythm_combo, params.cardiac_rhythm),
                             (self.resp_combo, params.resp_pattern)):
            blocked = combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(blocked)
        blocked = self.alarm_check.blockSignals(True)
        self.alarm_check.setChecked(params.alarm_enabled)
        self.alarm_check.blockSignals(blocked)
        # The combos were updated with their signals blocked to avoid writing
        # straight back to state, which also suppresses the summary's own
        # currentTextChanged hook - so refresh it explicitly.
        self._refresh_summary()


class AlarmBanner(QLabel):
    """Therapy alarm strip.

    Lives above the monitor rather than in the control panel: an alarm that
    disappears when the panel is collapsed is worse than no alarm at all.
    Arrest rhythms flash; unstable-but-perfusing rhythms are steady, so the two
    are distinguishable at a glance without reading the text.
    """

    LETHAL = ("#4a0f16", "#7a1a25", "#ff8a94")     # dim bg, bright bg, text
    URGENT = ("#3a2a0d", "#3a2a0d", "#e0a44c")     # steady

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("alarm")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._palette = self.URGENT
        self._bright = False
        self._active = False
        self._headline = ""
        self.clear_alarm()

    @property
    def is_active(self) -> bool:
        """Whether an alarm is raised.

        Not the same as ``isVisible()``: a widget in a window that has not been
        shown reports False regardless, which would make the banner rebuild
        itself every tick instead of flashing.
        """
        return self._active

    @property
    def headline(self) -> str:
        return self._headline

    def clear_alarm(self) -> None:
        self._active = False
        self._headline = ""
        self.setText("")
        self.setVisible(False)

    def show_alarm(self, headline: str, advice: str, urgency: str) -> None:
        self._palette = self.LETHAL if urgency == URGENCY_LETHAL else self.URGENT
        self._active = True
        self._headline = headline
        self.setText(f"⚠  {headline}  —  {advice}")
        self.setVisible(True)
        self._repaint()

    def flash(self) -> None:
        """Advance the flash phase; a no-op for the steady (urgent) palette."""
        self._bright = not self._bright
        self._repaint()

    def _repaint(self) -> None:
        dim, bright, text = self._palette
        background = bright if self._bright else dim
        self.setStyleSheet(f"QLabel#alarm {{ background: {background}; color: {text}; }}")


class MainWindow(QMainWindow):
    """Control panel and monitor, split 1:3."""

    def __init__(self, defaults: StateSnapshot | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ECG Simulator")
        self.setStyleSheet(STYLESHEET)
        self.resize(1400, 820)

        self.controls = ControlPanel(defaults)
        self.controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # The panel is taller than a short window once the pathology summary is
        # in it, so it scrolls rather than clipping its own bottom controls.
        self.controls_pane = QScrollArea(objectName="panel")
        self.controls_pane.setWidget(self.controls)
        self.controls_pane.setWidgetResizable(True)
        self.controls_pane.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_pane.setMinimumWidth(240)
        self.controls_pane.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # The toggle lives outside the panel: put it inside and hiding the panel
        # would hide the only way to bring it back.
        self.toggle_button = QPushButton(COLLAPSE_GLYPH, objectName="rail")
        self.toggle_button.setFixedWidth(18)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Fixed,
                                         QSizePolicy.Policy.Expanding)
        self.toggle_button.setToolTip("Hide the control panel (Ctrl+H)")
        self.toggle_button.clicked.connect(self.toggle_controls)

        self.monitor = MonitorView()
        self.alarm_banner = AlarmBanner()

        display = QWidget(objectName="root")
        display_layout = QVBoxLayout(display)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(0)
        display_layout.addWidget(self.alarm_banner)
        display_layout.addWidget(self.monitor, stretch=1)

        root = QWidget(objectName="root")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.controls_pane, stretch=1)
        layout.addWidget(self.toggle_button)
        layout.addWidget(display, stretch=3)
        self.setCentralWidget(root)

        # One timer drives the alarm: it re-reads the rhythm (which a shock can
        # change from the generator thread) and advances the flash phase.
        self._alarm_timer = QTimer(self)
        self._alarm_timer.timeout.connect(self._refresh_alarm)
        self._alarm_timer.start(ALARM_FLASH_MS)

        for sequence in ("Ctrl+H", "F9"):
            QShortcut(QKeySequence(sequence), self, activated=self.toggle_controls)

        self._state = None
        self._buffer = None
        self._generator = None
        # Swapped out in tests so the export path can be supplied without a dialog.
        self.choose_export_path = self._ask_export_path

    # -- control panel visibility ---------------------------------------------
    @property
    def controls_visible(self) -> bool:
        return not self.controls_pane.isHidden()

    def set_controls_visible(self, visible: bool) -> None:
        """Show or collapse the left pane; the monitor takes the freed width."""
        self.controls_pane.setVisible(visible)
        self.toggle_button.setText(COLLAPSE_GLYPH if visible else EXPAND_GLYPH)
        self.toggle_button.setToolTip(
            "Hide the control panel (Ctrl+H)" if visible
            else "Show the control panel (Ctrl+H)")

    def toggle_controls(self) -> None:
        self.set_controls_visible(not self.controls_visible)

    # -- wiring ---------------------------------------------------------------
    def bind(self, state, buffer, generator=None) -> None:
        """Connect the panel to the simulation and start the sweep."""
        self._state = state
        self._buffer = buffer
        self._generator = generator

        self.controls.parameter_changed.connect(self._on_parameter_changed)
        self.controls.motion_requested.connect(state.trigger_motion)
        self.controls.pvc_requested.connect(state.trigger_pvc)
        self.controls.export_requested.connect(self.export_buffer)
        self.controls.shock_requested.connect(self.deliver_shock)
        self.controls.alarm_toggled.connect(
            lambda on: self._state.update(alarm_enabled=on))
        if generator is not None:
            generator.shock_delivered.connect(self._on_shock_delivered)
        self.controls.rhythm_combo.currentTextChanged.connect(
            lambda text: self._state.update(cardiac_rhythm=text)
        )
        self.controls.resp_combo.currentTextChanged.connect(
            lambda text: self._state.update(resp_pattern=text)
        )
        self.controls.sync(state.snapshot())

        self.monitor.attach(buffer)
        self.monitor.start()
        self._refresh_alarm()
        self.controls.set_status(f"Sweep running - {buffer.duration:.0f} s window.")

    def _on_parameter_changed(self, key: str, value: float) -> None:
        # The panel emits state field names, so this stays a single hop no
        # matter how many sliders the panel grows.
        self._state.update(**{key: value})

    # -- therapy and alarm ----------------------------------------------------
    def deliver_shock(self, kind: str) -> None:
        """Queue one shock; the generator applies it on its next chunk."""
        if self._state is None:
            return
        if kind not in SHOCK_KINDS:
            raise ValueError(f"unknown shock kind {kind!r}")
        self._state.trigger_shock(kind)
        self.controls.set_status(
            "Unsynchronised shock charging..." if kind == THERAPY_DEFIB
            else "Synchronising to the R wave...")

    def _on_shock_delivered(self, kind: str, outcome: str, rhythm: str) -> None:
        self.controls.set_status(SHOCK_MESSAGES.get(outcome, outcome).format(kind=kind))
        self.controls.sync(self._state.snapshot())     # a shock can change the rhythm
        self._refresh_alarm()

    def _refresh_alarm(self) -> None:
        """Re-read the rhythm and update the banner.  Also drives the flashing."""
        if self._state is None:
            self.alarm_banner.clear_alarm()
            return
        params = self._state.snapshot()
        spec = therapy_for(params.cardiac_rhythm)
        if not params.alarm_enabled or not spec.alarm:
            self.alarm_banner.clear_alarm()
            return
        if self.alarm_banner.is_active and self.alarm_banner.headline == spec.alarm:
            self.alarm_banner.flash()
        else:
            self.alarm_banner.show_alarm(spec.alarm, spec.advice, spec.urgency)

    # -- export ---------------------------------------------------------------
    def _ask_export_path(self) -> str:
        default = f"ecg_sim_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export buffer to CSV", default, "CSV files (*.csv);;All files (*)"
        )
        return path

    def export_buffer(self) -> str | None:
        """Dump the current buffer.  Returns the path written, or None."""
        if self._buffer is None:
            return None
        path = self.choose_export_path()
        if not path:
            return None
        try:
            rows = self._buffer.to_csv(path)
        except OSError as exc:
            self.controls.set_status(f"Export failed: {exc}")
            return None
        self.controls.set_status(f"Exported {rows} samples to {os.path.basename(path)}")
        return path

    # -- lifecycle ------------------------------------------------------------
    def closeEvent(self, event) -> None:
        """Stop the sweep and join the producer before the window goes away.

        Without this the interpreter can tear down while the QThread is still
        writing into the buffer, which Qt reports as a crash on exit.
        """
        self.monitor.stop()
        if self._generator is not None:
            self._generator.stop()
        super().closeEvent(event)
