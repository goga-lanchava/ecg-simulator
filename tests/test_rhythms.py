"""Coverage for the full rhythm and breathing catalogue.

:mod:`tests.test_pathology` covers the original four in depth; this file checks
every entry in the catalogue and asserts the distinguishing feature of each new
one - the sawtooth of flutter, the lengthening PR of Wenckebach, the AV
dissociation of complete block, the twisting axis of torsades.

Rates are read from ``beat_log`` where peak detection is unreliable by design:
hyperkalaemia's peaked T waves and torsades' inverted complexes both defeat a
fixed-threshold R-peak detector, which is exactly what they do to a real monitor.

Run directly (``python tests/test_rhythms.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication                     # noqa: E402

from core.buffer import RingBuffer                            # noqa: E402
from core.pathology import (                                  # noqa: E402
    COMPLETE_BLOCK_ESCAPE_BPM,
    FLUTTER_RATE_HZ,
    MOBITZ_II_CONDUCTED,
    RESP_SPECS,
    RHYTHMS,
    WENCKEBACH_PR,
)
from core.state import (                                      # noqa: E402
    CARDIAC_RHYTHMS,
    RESP_PATTERNS,
    CardiacRhythm,
    RespPattern,
    SimulationState,
)
from tests.test_generator import (                            # noqa: E402
    FS,
    band_power,
    build,
    fwhm,
    r_peaks,
    run_for,
)
from ui.main_window import MainWindow                         # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv[:1])

# Rhythms with no organised complexes: rate assertions do not apply.
NON_BEAT = ("V-Fib", "Asystole")


def scheduled(engine, kinds=("sinus", "pvc", "escape")) -> list[float]:
    """Beat times from the log, by kind - independent of peak detection."""
    return sorted(t for t, kind in engine.beat_log if kind in kinds)


def scheduled_rate(engine, kinds=("sinus", "pvc", "escape")) -> float:
    times = scheduled(engine, kinds)
    return 60.0 / np.diff(times).mean()


# --- catalogue integrity ----------------------------------------------------
def test_every_rhythm_generates_a_finite_signal():
    """Nothing in the catalogue crashes, stalls, or emits NaN/inf."""
    for rhythm in CARDIAC_RHYTHMS:
        ecg, resp = run_for(build(heart_rate=72, cardiac_rhythm=rhythm), 12.0)
        assert np.isfinite(ecg).all(), f"{rhythm}: non-finite ECG"
        assert np.isfinite(resp).all(), f"{rhythm}: non-finite respiration"
        assert np.abs(ecg).max() < 10.0, f"{rhythm}: runaway amplitude"


def test_every_breathing_pattern_generates_a_finite_signal():
    for pattern in RESP_PATTERNS:
        _, resp = run_for(build(resp_pattern=pattern), 90.0)
        assert np.isfinite(resp).all(), f"{pattern}: non-finite respiration"
        assert np.abs(resp).max() < 3.0, f"{pattern}: runaway amplitude"


def test_literals_match_the_catalogue():
    """The type aliases and the runtime tuples must not drift apart."""
    from typing import get_args
    assert get_args(CardiacRhythm) == CARDIAC_RHYTHMS == tuple(RHYTHMS)
    assert get_args(RespPattern) == RESP_PATTERNS == tuple(RESP_SPECS)


def test_every_rhythm_is_reachable_from_the_ui():
    window = MainWindow()
    try:
        combo = window.controls.rhythm_combo
        listed = tuple(combo.itemText(i) for i in range(combo.count()))
        assert listed == CARDIAC_RHYTHMS
        breathing = window.controls.resp_combo
        assert tuple(breathing.itemText(i)
                     for i in range(breathing.count())) == RESP_PATTERNS
    finally:
        window.close()


def test_rate_overriding_rhythms_ignore_the_slider():
    """A rhythm with its own pacemaker must not follow the heart-rate control."""
    for rhythm, spec in RHYTHMS.items():
        if spec.rate_bpm is None or rhythm in NON_BEAT or spec.scheduler != "regular":
            continue
        for slider in (30, 200):
            engine = build(heart_rate=slider, cardiac_rhythm=rhythm)
            run_for(engine, 20.0)
            measured = scheduled_rate(engine)
            assert abs(measured - spec.rate_bpm) < 3, (
                f"{rhythm} at slider {slider}: {measured:.0f} bpm, "
                f"expected {spec.rate_bpm:.0f}")


def test_slider_following_rhythms_track_the_control():
    for rhythm, spec in RHYTHMS.items():
        if spec.rate_bpm is not None or spec.scheduler != "regular":
            continue
        engine = build(heart_rate=95, cardiac_rhythm=rhythm)
        run_for(engine, 20.0)
        measured = scheduled_rate(engine)
        assert abs(measured - 95) / 95 < 0.06, f"{rhythm}: {measured:.0f} bpm"


# --- atrial rhythms ---------------------------------------------------------
def test_atrial_flutter_has_sawtooth_waves_at_300_per_minute():
    flutter, _ = run_for(build(cardiac_rhythm="Atrial Flutter"), 20.0)
    sinus, _ = run_for(build(cardiac_rhythm="Sinus"), 20.0)

    at_5hz = band_power(flutter, FLUTTER_RATE_HZ - 0.1, FLUTTER_RATE_HZ + 0.1)
    assert at_5hz > 5 * band_power(sinus, FLUTTER_RATE_HZ - 0.1, FLUTTER_RATE_HZ + 0.1)
    # A sawtooth is rich in harmonics; a plain sine would not be.
    assert band_power(flutter, 9.9, 10.1) > 5 * band_power(sinus, 9.9, 10.1)


def test_atrial_flutter_conducts_two_to_one():
    engine = build(heart_rate=72, cardiac_rhythm="Atrial Flutter")
    run_for(engine, 20.0)
    ventricular = scheduled_rate(engine)
    atrial = FLUTTER_RATE_HZ * 60.0
    assert abs(ventricular - 150.0) < 3
    assert abs(atrial / ventricular - 2.0) < 0.05, "not 2:1 conduction"


def test_svt_is_fast_and_narrow():
    ecg, _ = run_for(build(cardiac_rhythm="SVT"), 20.0)
    peaks = r_peaks(ecg)
    assert abs(60.0 / (np.diff(peaks).mean() / FS) - 180) < 4
    assert fwhm(ecg, peaks[4]) < 0.040, "SVT complexes should stay narrow"


def test_junctional_is_slow_with_a_retrograde_p_wave():
    engine = build(cardiac_rhythm="Junctional")
    ecg, _ = run_for(engine, 30.0)
    assert abs(scheduled_rate(engine) - 48) < 2

    peak = r_peaks(ecg)[4]
    base = float(np.median(ecg[peak - 400: peak - 350]))
    # The P is inverted and sits close to the QRS instead of 200 ms ahead.
    assert ecg[peak - 100: peak - 60].min() - base < -0.03, "P wave is not inverted"


# --- conduction blocks ------------------------------------------------------
def test_first_degree_block_has_a_long_but_constant_pr():
    ecg, _ = run_for(build(heart_rate=60, cardiac_rhythm="1st Degree AV Block"), 20.0)
    offsets = []
    for peak in r_peaks(ecg)[2:8]:
        window = ecg[peak - 450: peak - 150]
        offsets.append(450 - int(np.argmax(window)))       # ms before the R peak
    offsets = np.array(offsets)
    assert abs(offsets.mean() - 320) < 40, f"PR measured {offsets.mean():.0f} ms"
    assert offsets.std() < 15, "PR should be constant in first-degree block"


def test_wenckebach_drops_a_beat_after_a_lengthening_pr():
    engine = build(heart_rate=72, cardiac_rhythm="Mobitz I (Wenckebach)")
    run_for(engine, 40.0)

    dropped = [t for t, kind in engine.beat_log if kind == "p"]
    conducted = [t for t, kind in engine.beat_log if kind == "sinus"]
    ratio = len(conducted) / (len(conducted) + len(dropped))
    expected = len(WENCKEBACH_PR) / (len(WENCKEBACH_PR) + 1)
    assert abs(ratio - expected) < 0.05, f"conducted {ratio:.2f}, expected {expected:.2f}"

    # Grouped beating: the PR increment shrinks across the cycle, so each R-R is
    # shorter than the one before it, and then comes the pause.
    rr = np.diff(sorted(conducted))
    pauses = np.flatnonzero(rr > 1.4 * np.median(rr))
    assert len(pauses) >= 3, "no dropped beats found in the R-R series"
    before = np.array([[rr[i - 2], rr[i - 1]] for i in pauses if i >= 2])
    assert before[:, 1].mean() < before[:, 0].mean(), (
        f"R-R does not shorten before the drop ({before.mean(axis=0)})")


def test_mobitz_ii_drops_a_beat_with_no_warning():
    engine = build(heart_rate=72, cardiac_rhythm="Mobitz II")
    ecg, _ = run_for(engine, 40.0)

    dropped = [t for t, kind in engine.beat_log if kind == "p"]
    conducted = [t for t, kind in engine.beat_log if kind == "sinus"]
    assert abs(len(conducted) / len(dropped) - MOBITZ_II_CONDUCTED) < 0.4

    # Unlike Wenckebach, the PR interval never moves.
    offsets = []
    for peak in r_peaks(ecg)[2:10]:
        offsets.append(300 - int(np.argmax(ecg[peak - 300: peak - 100])))
    assert np.std(offsets) < 12, "PR should be constant in Mobitz II"

    # The pause is a clean multiple of the underlying P-P interval.
    rr = np.diff(sorted(conducted))
    assert rr.max() > 1.7 * np.median(rr), "no dropped beat visible in the R-R series"


def test_complete_block_dissociates_atria_from_ventricles():
    engine = build(heart_rate=90, cardiac_rhythm="3rd Degree AV Block")
    run_for(engine, 40.0)

    p_times = np.array(sorted(t for t, kind in engine.beat_log if kind == "p"))
    v_times = np.array(sorted(t for t, kind in engine.beat_log if kind == "escape"))
    atrial = 60.0 / np.diff(p_times).mean()
    ventricular = 60.0 / np.diff(v_times).mean()

    assert abs(atrial - 90) < 3, f"atrial rate {atrial:.0f}, slider said 90"
    assert abs(ventricular - COMPLETE_BLOCK_ESCAPE_BPM) < 2
    assert atrial > ventricular * 1.8, "rates are not dissociated"

    # No fixed relationship: each QRS falls at a different phase of the atrial
    # cycle, so the phases spread out like uniform noise (std -> 0.289).
    pp = np.diff(p_times).mean()
    phases = [(v - p_times[p_times <= v][-1]) / pp
              for v in v_times[2:-2] if (p_times <= v).any()]
    assert np.std(phases) > 0.2, f"P and QRS look coupled (phase std {np.std(phases):.3f})"


# --- ventricular rhythms ----------------------------------------------------
def test_bigeminy_alternates_narrow_and_wide_complexes():
    engine = build(heart_rate=72, cardiac_rhythm="PVC Bigeminy")
    ecg, _ = run_for(engine, 30.0)

    kinds = [kind for _, kind in engine.beat_log]
    assert kinds[2:10] == ["pvc", "sinus"] * 4 or kinds[2:10] == ["sinus", "pvc"] * 4

    peaks = r_peaks(ecg)[2:12]
    widths = np.array([fwhm(ecg, p) for p in peaks])
    narrow, wide = widths[widths < 0.05], widths[widths >= 0.05]
    assert len(narrow) >= 3 and len(wide) >= 3, "complexes do not alternate in width"
    assert wide.mean() > 2.5 * narrow.mean()


def test_torsades_twists_about_the_baseline():
    engine = build(cardiac_rhythm="Torsades de Pointes")
    ecg, _ = run_for(engine, 30.0)

    assert abs(scheduled_rate(engine) - RHYTHMS["Torsades de Pointes"].rate_bpm) < 6

    # The envelope waxes and wanes: peak amplitude in half-second windows varies
    # far more than it does for monomorphic V-Tach.
    def envelope_spread(signal):
        blocks = signal[: signal.size // 500 * 500].reshape(-1, 500)
        peaks = np.abs(blocks).max(axis=1)
        return peaks.std() / peaks.mean()

    vtach, _ = run_for(build(cardiac_rhythm="V-Tach"), 30.0)
    assert envelope_spread(ecg) > 3 * envelope_spread(vtach), "amplitude does not twist"

    # ...and the complexes invert, which monomorphic VT never does.
    assert ecg.min() < -0.5 and ecg.max() > 0.5


def test_vfib_has_no_identifiable_complexes():
    ecg, _ = run_for(build(cardiac_rhythm="V-Fib"), 20.0)
    peaks, _ = find_peaks(ecg, height=0.8, distance=int(0.2 * FS))
    assert len(peaks) == 0, "V-Fib should show no QRS-like complexes"

    assert np.ptp(ecg) > 0.4, "V-Fib is not just a flat line"
    # Continuous activity: VF waxes and wanes from coarse to fine, but never
    # goes isoelectric - that would be asystole, which is a different rhythm.
    quiet = [np.ptp(ecg[i: i + 300]) for i in range(0, ecg.size - 300, 300)]
    assert min(quiet) > 0.05, f"V-Fib has an isoelectric pause ({min(quiet):.3f})"
    flat, _ = run_for(build(cardiac_rhythm="Asystole"), 5.0)
    assert min(quiet) > 100 * max(np.ptp(flat), 1e-9)


def test_asystole_is_a_flat_line():
    ecg, _ = run_for(build(cardiac_rhythm="Asystole"), 20.0)
    assert np.ptp(ecg) < 1e-9, f"asystole is not flat (ptp {np.ptp(ecg):.4f})"

    # ...but the artifact pipeline still reaches it, so a lead can still be knocked.
    engine = build(cardiac_rhythm="Asystole")
    run_for(engine, 1.0)
    engine.state.trigger_motion()
    during, _ = run_for(engine, 2.0)
    assert np.abs(during).max() > 1.5


def test_idioventricular_is_slow_and_wide():
    engine = build(heart_rate=72, cardiac_rhythm="Idioventricular")
    ecg, _ = run_for(engine, 30.0)
    assert abs(scheduled_rate(engine) - 34) < 2, "not a slow escape rhythm"
    assert fwhm(ecg, r_peaks(ecg)[3]) > 0.06, "complexes are not broad"


# --- ischaemia and metabolic ------------------------------------------------
def test_ischemia_depresses_st_and_inverts_the_t_wave():
    from tests.test_pathology import PRE, ensemble

    ecg, _ = run_for(build(heart_rate=72, cardiac_rhythm="Ischemia (ST Dep.)"), 30.0)
    average, base = ensemble(ecg, r_peaks(ecg))

    st = average[PRE + 70] - base
    t_wave = average[PRE + 200: PRE + 400].min() - base
    assert st < -0.15, f"ST is not depressed ({st:+.3f})"
    assert t_wave < -0.15, f"T wave is not inverted ({t_wave:+.3f})"


def test_hyperkalemia_has_tall_peaked_t_waves():
    from tests.test_pathology import PRE, ensemble

    engine = build(heart_rate=72, cardiac_rhythm="Hyperkalemia")
    ecg, _ = run_for(engine, 30.0)
    # Detect on R only: the peaked T would otherwise be counted as a complex.
    peaks, _ = find_peaks(ecg, height=0.9, distance=int(0.3 * FS))
    average, base = ensemble(ecg, peaks)

    t_amplitude = average[PRE + 120: PRE + 400].max() - base
    assert t_amplitude > 0.6, f"T wave is not tall ({t_amplitude:.2f})"

    # "Peaked" means narrow as well as tall - much narrower than a sinus T.
    t_index = PRE + 120 + int(np.argmax(average[PRE + 120: PRE + 400]))
    half = base + (average[t_index] - base) / 2
    width = np.sum(average[PRE + 100: PRE + 400] > half) / FS
    assert width < 0.10, f"T wave is broad, not peaked ({width * 1000:.0f} ms)"

    assert abs(scheduled_rate(engine) - 72) < 3, "rate should still follow the slider"


def test_paced_beats_carry_a_pacing_spike():
    """A stimulus artifact is the sharpest feature on the trace.

    Measured spectrally rather than by width: the paced QRS is deliberately
    broad, so a threshold crossing near the spike also catches the QRS upstroke.
    Idioventricular is the fair control - equally wide, but not paced.  The band
    tracks PACING_SPIKE_WIDTH: a display-width spike concentrates its energy
    around 20-60 Hz rather than above 80 Hz.
    """
    paced, _ = run_for(build(cardiac_rhythm="Paced"), 20.0)
    unpaced, _ = run_for(build(cardiac_rhythm="Idioventricular"), 20.0)

    assert band_power(paced, 20, 60) > 20 * band_power(unpaced, 20, 60), (
        "no sharp stimulus artifact present")

    # And it sits just before each complex, not on top of it.
    peaks, _ = find_peaks(paced, height=0.9, distance=int(0.3 * FS))
    spikes = sum(1 for peak in peaks[2:8] if paced[peak - 90: peak - 40].max() > 0.3)
    assert spikes >= 5, f"only {spikes}/6 paced beats had a spike before the QRS"


# --- breathing patterns -----------------------------------------------------
def test_kussmaul_is_deep_and_fast():
    _, kussmaul = run_for(build(respiratory_rate=15, resp_pattern="Kussmaul"), 60.0)
    _, normal = run_for(build(respiratory_rate=15), 60.0)

    def rate(x):
        peaks, _ = find_peaks(x, distance=int(0.5 * FS), height=0.3)
        return 60.0 / (np.diff(peaks).mean() / FS)

    assert rate(kussmaul) > 1.5 * rate(normal), "Kussmaul is not fast"
    assert np.abs(kussmaul).max() > 1.25 * np.abs(normal).max(), "not deep"


def test_apnoea_is_total_cessation():
    _, resp = run_for(build(resp_pattern="Apnoea"), 60.0)
    assert np.all(resp == 0.0), "apnoea still shows respiratory effort"


def test_agonal_is_slow_infrequent_gasps():
    _, resp = run_for(build(respiratory_rate=15, resp_pattern="Agonal"), 120.0)
    peaks, _ = find_peaks(resp, distance=int(2.0 * FS), height=0.3)
    rate = 60.0 / (np.diff(peaks).mean() / FS)
    assert abs(rate - 6.0) < 1.0, f"agonal rate {rate:.1f}, expected ~6"

    # A gasp is a sharp inspiration followed by a long collapse.
    rising = np.sum(np.diff(resp) > 0)
    assert rising / resp.size < 0.25, "waveform is not gasp-shaped"


def test_biot_breathes_in_irregular_clusters():
    _, resp = run_for(build(resp_pattern="Biot (Ataxic)"), 180.0)

    silent = np.abs(resp) < 1e-9
    assert silent.mean() > 0.2, "no pauses at all"
    assert silent.mean() < 0.8, "nothing but pauses"

    # Cluster lengths must actually vary - that is what makes it ataxic.
    edges = np.diff(silent.astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = np.diff(np.flatnonzero(edges != 0))
    assert len(starts) >= 3 and len(stops) >= 3, "too few clusters to judge"
    assert np.std(runs) / np.mean(runs) > 0.15, "clusters are too regular for Biot"


def test_breathing_patterns_are_independent_of_rhythm():
    """A pathology on one channel must not disturb the other."""
    _, with_vtach = run_for(build(cardiac_rhythm="V-Tach", resp_pattern="Kussmaul"), 30.0)
    _, alone = run_for(build(resp_pattern="Kussmaul"), 30.0)
    assert np.allclose(with_vtach, alone, atol=1e-9)


# --- switching --------------------------------------------------------------
def test_switching_between_all_rhythms_stays_stable():
    """Cycle the whole catalogue on one engine: no bursts, no stalls, no NaN."""
    engine = build(heart_rate=72)
    for rhythm in CARDIAC_RHYTHMS:
        engine.state.update(cardiac_rhythm=rhythm)
        ecg, _ = run_for(engine, 3.0)
        assert np.isfinite(ecg).all(), f"{rhythm}: non-finite after switch"
        assert np.abs(ecg).max() < 10.0, f"{rhythm}: amplitude spike after switch"

    # Back to sinus, the rate must recover rather than stay stuck.
    engine.state.update(cardiac_rhythm="Sinus")
    run_for(engine, 2.0)
    ecg, _ = run_for(engine, 20.0)
    measured = 60.0 / (np.diff(r_peaks(ecg)).mean() / FS)
    assert abs(measured - 72) < 5, f"after cycling every rhythm: {measured:.0f} bpm"


def test_switching_never_back_dates_a_burst_of_beats():
    """A slow rhythm followed by a fast one must not dump a catch-up volley."""
    engine = build(heart_rate=72, cardiac_rhythm="Idioventricular")
    run_for(engine, 6.0)
    engine.state.update(cardiac_rhythm="SVT")
    ecg, _ = run_for(engine, 6.0)

    peaks = r_peaks(ecg)
    if len(peaks) > 2:
        shortest = np.diff(peaks).min() / FS
        assert shortest > 0.20, f"beats piled up: shortest R-R {shortest * 1000:.0f} ms"


def test_beat_log_is_bounded():
    """A monitor may run for hours; the diagnostic log must not grow forever."""
    from core.generator import BEAT_LOG_LIMIT

    engine = build(heart_rate=200, cardiac_rhythm="V-Tach")
    run_for(engine, 200.0)
    assert len(engine.beat_log) <= BEAT_LOG_LIMIT


def test_dropdowns_drive_every_option_end_to_end():
    state = SimulationState()
    window = MainWindow(state.snapshot())
    window.bind(state, RingBuffer())
    try:
        for rhythm in CARDIAC_RHYTHMS:
            window.controls.rhythm_combo.setCurrentText(rhythm)
            assert state.snapshot().cardiac_rhythm == rhythm
        for pattern in RESP_PATTERNS:
            window.controls.resp_combo.setCurrentText(pattern)
            assert state.snapshot().resp_pattern == pattern
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
