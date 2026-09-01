"""Defibrillation, synchronised cardioversion, and the therapy alarm.

The outcomes are drawn from a distribution, so the statistical assertions run
many seeded trials rather than one.  What matters is not the exact success rate
but that the clinically important distinctions hold every single time:

  * asystole is never shockable, however many times you press the button;
  * a synchronised shock cannot be delivered without an R wave to time it to;
  * an unsynchronised shock into an organised rhythm can induce VF.

Run directly (``python tests/test_therapy.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication                       # noqa: E402

from core.buffer import RingBuffer                              # noqa: E402
from core.generator import (                                    # noqa: E402
    CARDIOVERT_SUCCESS,
    DEFIB_SUCCESS,
    POST_SHOCK_PAUSE_S,
    R_ON_T_RISK,
    SHOCK_CONVERTED,
    SHOCK_INDUCED_VF,
    SHOCK_NOT_DELIVERED,
    SHOCK_NO_EFFECT,
    SHOCK_PERSISTS,
    SHOCK_SPIKE_MV,
    SignalGenerator,
)
from core.pathology import (                                    # noqa: E402
    SHOCK_KINDS,
    THERAPIES,
    THERAPY_CARDIOVERT,
    THERAPY_CPR,
    THERAPY_DEFIB,
    UNSYNCHRONISABLE,
    URGENCY_LETHAL,
    therapy_for,
)
from core.state import CARDIAC_RHYTHMS, SimulationState         # noqa: E402
from tests.test_generator import FS, build, r_peaks, run_for    # noqa: E402
from ui.main_window import MainWindow                           # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv[:1])


def shock(rhythm: str, kind: str, seed: int = 0, settle: float = 2.0,
          after: float = 4.0):
    """Run one engine, deliver one shock, return (engine, outcome, new rhythm)."""
    engine = build(seed=seed, cardiac_rhythm=rhythm)
    run_for(engine, settle)
    engine.state.trigger_shock(kind)
    ecg, _ = run_for(engine, after)
    _, _, outcome, resulting = engine.shock_log[-1]
    return engine, outcome, resulting, ecg


def outcomes(rhythm: str, kind: str, trials: int = 60) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in range(trials):
        _, outcome, _, _ = shock(rhythm, kind, seed=seed)
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


# --- the indication table ---------------------------------------------------
def test_every_therapy_entry_names_a_real_rhythm():
    assert not set(THERAPIES) - set(CARDIAC_RHYTHMS)


def test_arrest_rhythms_are_flagged_lethal():
    for rhythm in ("V-Fib", "V-Tach", "Torsades de Pointes", "Asystole"):
        assert therapy_for(rhythm).urgency == URGENCY_LETHAL, rhythm
        assert therapy_for(rhythm).alarm, f"{rhythm} has no alarm text"


def test_stable_rhythms_carry_no_indication():
    for rhythm in ("Sinus", "1st Degree AV Block", "STEMI", "Paced"):
        assert therapy_for(rhythm).therapy == "", rhythm
        assert therapy_for(rhythm).alarm == ""


def test_asystole_is_marked_not_shockable():
    assert therapy_for("Asystole").therapy == THERAPY_CPR
    assert "NOT SHOCKABLE" in therapy_for("Asystole").alarm


# --- outcomes ---------------------------------------------------------------
def test_defibrillation_converts_ventricular_fibrillation():
    counts = outcomes("V-Fib", THERAPY_DEFIB)
    assert set(counts) <= {SHOCK_CONVERTED, SHOCK_PERSISTS}
    rate = counts.get(SHOCK_CONVERTED, 0) / sum(counts.values())
    assert abs(rate - DEFIB_SUCCESS) < 0.18, f"success rate {rate:.2f}"
    assert counts.get(SHOCK_PERSISTS, 0) > 0, "a shock that never fails is not honest"


def test_cardioversion_converts_atrial_fibrillation():
    counts = outcomes("AFib", THERAPY_CARDIOVERT)
    assert set(counts) <= {SHOCK_CONVERTED, SHOCK_PERSISTS}
    rate = counts.get(SHOCK_CONVERTED, 0) / sum(counts.values())
    assert abs(rate - CARDIOVERT_SUCCESS) < 0.15, f"success rate {rate:.2f}"


def test_asystole_never_responds_to_a_shock():
    """The classic error: shocking a flat line. It must never 'work'."""
    for kind in SHOCK_KINDS:
        counts = outcomes("Asystole", kind, trials=40)
        assert SHOCK_CONVERTED not in counts, f"{kind} converted asystole"
        assert set(counts) <= {SHOCK_NO_EFFECT, SHOCK_NOT_DELIVERED}, counts

    _, _, resulting, _ = shock("Asystole", THERAPY_DEFIB)
    assert resulting == "Asystole", "asystole changed after a shock"


def test_cardioversion_cannot_synchronise_without_an_r_wave():
    for rhythm in UNSYNCHRONISABLE:
        counts = outcomes(rhythm, THERAPY_CARDIOVERT, trials=25)
        assert counts == {SHOCK_NOT_DELIVERED: 25}, f"{rhythm}: {counts}"


def test_unsynchronised_shock_on_a_perfusing_rhythm_can_induce_vf():
    """R-on-T is the reason cardioversion is synchronised at all."""
    counts = outcomes("Sinus", THERAPY_DEFIB)
    induced = counts.get(SHOCK_INDUCED_VF, 0) / sum(counts.values())
    assert abs(induced - R_ON_T_RISK) < 0.18, f"induced VF in {induced:.2f}"
    assert set(counts) <= {SHOCK_NO_EFFECT, SHOCK_INDUCED_VF}
    assert SHOCK_CONVERTED not in counts, "a shock 'converted' normal sinus rhythm"


def test_synchronised_shock_is_safe_on_a_perfusing_rhythm():
    counts = outcomes("Sinus", THERAPY_CARDIOVERT, trials=40)
    assert counts == {SHOCK_NO_EFFECT: 40}, counts


def test_induced_vf_actually_leaves_the_patient_in_vf():
    for seed in range(60):
        engine, outcome, resulting, _ = shock("Sinus", THERAPY_DEFIB, seed=seed)
        if outcome == SHOCK_INDUCED_VF:
            assert resulting == "V-Fib"
            assert engine.state.snapshot().cardiac_rhythm == "V-Fib"
            return
    raise AssertionError("never induced VF across 60 trials")


def test_conversion_puts_the_patient_into_sinus():
    for seed in range(40):
        engine, outcome, resulting, _ = shock("V-Fib", THERAPY_DEFIB, seed=seed)
        if outcome == SHOCK_CONVERTED:
            assert resulting == "Sinus"
            assert engine.state.snapshot().cardiac_rhythm == "Sinus"
            return
    raise AssertionError("never converted across 40 trials")


def test_every_rhythm_survives_both_shocks():
    """No rhythm may crash, hang or emit non-finite signal when shocked."""
    for rhythm in CARDIAC_RHYTHMS:
        for kind in SHOCK_KINDS:
            _, outcome, _, ecg = shock(rhythm, kind, seed=1, after=3.0)
            assert np.isfinite(ecg).all(), f"{rhythm}+{kind}: non-finite output"
            assert outcome in (SHOCK_CONVERTED, SHOCK_PERSISTS, SHOCK_NO_EFFECT,
                               SHOCK_INDUCED_VF, SHOCK_NOT_DELIVERED), outcome


# --- the artifact -----------------------------------------------------------
def test_the_shock_blanks_the_trace():
    _, _, _, ecg = shock("V-Fib", THERAPY_DEFIB, seed=0, settle=2.0, after=4.0)
    assert ecg.max() > 0.8 * SHOCK_SPIKE_MV, "no discharge artifact"

    # Saturation recovers: the tail is back inside the normal display range.
    assert np.abs(ecg[-1000:]).max() < 2.5, "amplifier never recovered"


def test_a_refused_shock_leaves_no_artifact():
    """If the defibrillator will not fire, nothing should appear on the trace."""
    _, outcome, _, ecg = shock("V-Fib", THERAPY_CARDIOVERT, seed=0)
    assert outcome == SHOCK_NOT_DELIVERED
    assert ecg.max() < 2.0, "an undelivered shock still drew an artifact"


def test_the_heart_pauses_after_a_shock():
    """Stunned myocardium: no complexes for a moment, then the new rhythm."""
    for seed in range(40):
        engine, outcome, _, ecg = shock("V-Fib", THERAPY_DEFIB, seed=seed,
                                        settle=2.0, after=6.0)
        if outcome != SHOCK_CONVERTED:
            continue
        beats = [t for t, kind in engine.beat_log if kind == "sinus"]
        after_shock = [t for t in beats if t > 2.0]
        assert after_shock, "no rhythm returned after conversion"
        assert min(after_shock) >= 2.0 + POST_SHOCK_PAUSE_S - 0.05, "no post-shock pause"

        # ...and a real rhythm is running by the end of the record.
        assert len(r_peaks(ecg[-3000:])) >= 2, "sinus rhythm did not resume"
        return
    raise AssertionError("never converted across 40 trials")


def test_shocks_are_consumed_exactly_once():
    state = SimulationState()
    assert state.consume_shock() is None
    state.trigger_shock(THERAPY_DEFIB)
    state.trigger_shock(THERAPY_CARDIOVERT)          # replaces, does not queue two
    assert state.consume_shock() == THERAPY_CARDIOVERT
    assert state.consume_shock() is None


def test_one_click_is_one_shock():
    engine = build(cardiac_rhythm="V-Fib")
    run_for(engine, 2.0)
    engine.state.trigger_shock(THERAPY_DEFIB)
    run_for(engine, 5.0)
    assert len(engine.shock_log) == 1, f"{len(engine.shock_log)} shocks from one click"


# --- UI ---------------------------------------------------------------------
def bound(rhythm: str = "Sinus", **params):
    state = SimulationState(cardiac_rhythm=rhythm, **params)
    buffer = RingBuffer()
    generator = SignalGenerator(state, buffer, rng=np.random.default_rng(3))
    window = MainWindow(state.snapshot())
    window.bind(state, buffer, generator)
    return window, state, generator


def test_therapy_buttons_queue_the_right_shock():
    window, state, _ = bound("V-Fib")
    try:
        window.controls.defib_button.click()
        assert state.pending_shock == THERAPY_DEFIB
        state.consume_shock()

        window.controls.cardiovert_button.click()
        assert state.pending_shock == THERAPY_CARDIOVERT
    finally:
        window.close()


def test_shock_result_is_reported_and_the_panel_resyncs():
    window, state, generator = bound("V-Fib")
    try:
        window.controls.defib_button.click()
        for _ in range(60):
            generator.engine.next_chunk(50)
        _, kind, outcome, resulting = generator.engine.shock_log[-1]
        window._on_shock_delivered(kind, outcome, resulting)

        assert window.controls.status_label.text(), "no outcome reported"
        assert window.controls.rhythm_combo.currentText() == resulting, (
            "dropdown out of step with the rhythm after a shock")
    finally:
        window.close()


# --- the alarm --------------------------------------------------------------
def test_alarm_raises_on_every_rhythm_that_needs_therapy():
    window, state, _ = bound()
    try:
        for rhythm in CARDIAC_RHYTHMS:
            state.update(cardiac_rhythm=rhythm)
            window._refresh_alarm()
            expected = bool(therapy_for(rhythm).alarm)
            assert window.alarm_banner.is_active is expected, (
                f"{rhythm}: alarm active={window.alarm_banner.is_active}, "
                f"expected {expected}")
            if expected:
                assert therapy_for(rhythm).alarm in window.alarm_banner.text()
                assert therapy_for(rhythm).advice in window.alarm_banner.text()
    finally:
        window.close()


def test_alarm_toggle_silences_it():
    window, state, _ = bound("V-Fib")
    try:
        window._refresh_alarm()
        assert window.alarm_banner.is_active

        window.controls.alarm_check.setChecked(False)
        window._refresh_alarm()
        assert state.snapshot().alarm_enabled is False
        assert not window.alarm_banner.is_active, "alarm ignored the toggle"

        window.controls.alarm_check.setChecked(True)
        window._refresh_alarm()
        assert window.alarm_banner.is_active
    finally:
        window.close()


def test_alarm_survives_the_control_panel_being_hidden():
    """The alarm must not be inside the pane the user just collapsed."""
    window, _, _ = bound("V-Fib")
    window.resize(1400, 820)
    window.show()
    APP.processEvents()
    try:
        window.set_controls_visible(False)
        APP.processEvents()
        assert not window.controls_pane.isVisible()
        assert window.alarm_banner.isVisible(), "alarm vanished with the panel"
    finally:
        window.close()


def test_lethal_rhythms_flash_and_urgent_ones_do_not():
    window, state, _ = bound("V-Fib")
    try:
        window._refresh_alarm()
        lethal = {window.alarm_banner.styleSheet()}
        for _ in range(3):
            window._refresh_alarm()
            lethal.add(window.alarm_banner.styleSheet())
        assert len(lethal) > 1, "arrest alarm does not flash"

        state.update(cardiac_rhythm="AFib")
        window._refresh_alarm()
        urgent = {window.alarm_banner.styleSheet()}
        for _ in range(3):
            window._refresh_alarm()
            urgent.add(window.alarm_banner.styleSheet())
        assert len(urgent) == 1, "urgent alarm should be steady, not flashing"
    finally:
        window.close()


def test_alarm_clears_when_the_shock_works():
    window, state, generator = bound("V-Fib")
    try:
        window._refresh_alarm()
        assert window.alarm_banner.is_active

        for seed in range(20):
            generator.engine.rng = np.random.default_rng(seed)
            state.update(cardiac_rhythm="V-Fib")
            state.trigger_shock(THERAPY_DEFIB)
            for _ in range(40):
                generator.engine.next_chunk(50)
            _, kind, outcome, resulting = generator.engine.shock_log[-1]
            if outcome == SHOCK_CONVERTED:
                window._on_shock_delivered(kind, outcome, resulting)
                assert not window.alarm_banner.is_active, (
                    "alarm still showing after successful conversion")
                return
        raise AssertionError("never converted across 20 trials")
    finally:
        window.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
