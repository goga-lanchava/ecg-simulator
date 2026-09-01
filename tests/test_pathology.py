"""Pathology checks: AFib, STEMI, V-Tach and Cheyne-Stokes respiration.

Each rhythm is verified by the feature a clinician would look for - an absent
P wave, an elevated ST segment, a broad QRS at a fixed rate, an apnoeic pause -
measured back out of the generated signal.

P-wave amplitude is read off an *ensemble average* of R-aligned beats.  Averaging
cancels anything not locked to the QRS, which is exactly how the fibrillatory
baseline is separated from a real P wave on a physical recording.

Run directly (``python tests/test_pathology.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication                    # noqa: E402

from core.buffer import RingBuffer                           # noqa: E402
from core.state import CARDIAC_RHYTHMS, RESP_PATTERNS, SimulationState  # noqa: E402
from tests.test_generator import (                                    # noqa: E402
    FS,
    band_power,
    build,
    fwhm,
    r_peaks,
    run_for,
)
from ui.main_window import MainWindow                        # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv[:1])

PRE, POST = 350, 450        # ensemble window either side of the R peak


def ensemble(ecg: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, float]:
    """R-aligned average beat, plus its PR-segment baseline."""
    beats = [ecg[p - PRE: p + POST] for p in peaks if p > PRE and p + POST < ecg.size]
    assert len(beats) > 10, f"only {len(beats)} beats to average"
    average = np.mean(beats, axis=0)
    return average, float(np.median(average[PRE - 95: PRE - 78]))


def rr_intervals(ecg: np.ndarray) -> np.ndarray:
    return np.diff(r_peaks(ecg)) / FS


# --- Atrial fibrillation ----------------------------------------------------
def test_afib_has_no_p_wave():
    """The defining feature: atrial depolarisation is gone."""
    amplitudes = {}
    for rhythm in ("Sinus", "AFib"):
        ecg, _ = run_for(build(heart_rate=72, cardiac_rhythm=rhythm), 40.0)
        average, base = ensemble(ecg, r_peaks(ecg))
        amplitudes[rhythm] = average[PRE - 300: PRE - 130].max() - base

    assert amplitudes["Sinus"] > 0.12, f"sinus P wave missing ({amplitudes['Sinus']:.3f})"
    assert amplitudes["AFib"] < 0.07, f"AFib still shows a P wave ({amplitudes['AFib']:.3f})"


def test_afib_is_irregularly_irregular():
    sinus = rr_intervals(run_for(build(heart_rate=72), 40.0)[0])
    afib = rr_intervals(run_for(build(heart_rate=72, cardiac_rhythm="AFib"), 40.0)[0])

    cv = lambda x: x.std() / x.mean()          # noqa: E731
    assert cv(sinus) < 0.06, f"sinus is not regular (CV {cv(sinus):.3f})"
    assert cv(afib) > 0.12, f"AFib is not irregular enough (CV {cv(afib):.3f})"
    assert cv(afib) > 3 * cv(sinus)


def test_afib_mean_rate_still_follows_the_slider():
    """Jitter is symmetric, so the *average* rate must still track the control."""
    for bpm in (60, 110):
        ecg, _ = run_for(build(heart_rate=bpm, cardiac_rhythm="AFib"), 60.0)
        measured = 60.0 / rr_intervals(ecg).mean()
        assert abs(measured - bpm) / bpm < 0.05, f"commanded {bpm}, measured {measured:.1f}"


def test_afib_shows_fibrillatory_waves_on_the_baseline():
    """The flat TP segment of sinus rhythm is replaced by a coarse undulation."""
    ripple = {}
    for rhythm in ("Sinus", "AFib"):
        # 45 bpm leaves a long, genuinely empty TP segment to look at.
        ecg, _ = run_for(build(heart_rate=45, cardiac_rhythm=rhythm), 60.0)
        peaks = r_peaks(ecg)
        spans = [np.ptp(ecg[a + 600: a + 880])
                 for a, b in zip(peaks, peaks[1:]) if b - a > 1200]
        ripple[rhythm] = float(np.median(spans))

    assert ripple["Sinus"] < 0.03, f"sinus baseline is not flat ({ripple['Sinus']:.4f})"
    assert ripple["AFib"] > 0.05, f"no f-waves present ({ripple['AFib']:.4f})"
    assert ripple["AFib"] > 3 * ripple["Sinus"]


# --- STEMI ------------------------------------------------------------------
def test_stemi_elevates_the_st_segment():
    """ST elevation is the finding that makes this a STEMI rather than an NSTEMI."""
    st = {}
    for rhythm in ("Sinus", "STEMI"):
        ecg, _ = run_for(build(heart_rate=72, cardiac_rhythm=rhythm), 30.0)
        average, base = ensemble(ecg, r_peaks(ecg))
        st[rhythm] = average[PRE + 60] - base       # J point, 60 ms after R

    assert st["Sinus"] < 0.0, f"sinus ST is not isoelectric/depressed ({st['Sinus']:+.3f})"
    assert st["STEMI"] > 0.20, f"ST segment is not elevated ({st['STEMI']:+.3f})"


def test_stemi_has_a_tall_broad_t_wave():
    ecg, _ = run_for(build(heart_rate=72, cardiac_rhythm="STEMI"), 30.0)
    average, base = ensemble(ecg, r_peaks(ecg))
    t_amplitude = average[PRE + 120: PRE + 400].max() - base
    assert t_amplitude > 0.40, f"T wave is not hyperacute ({t_amplitude:.3f})"

    # "Tombstone": the ST segment never returns to baseline between S and T.
    trough = average[PRE + 60: PRE + 150].min() - base
    assert trough > 0.15, f"ST returns to baseline ({trough:+.3f}); no tombstone"


def test_stemi_keeps_a_normal_regular_rate():
    """Only the morphology changes - the rate must still obey the slider."""
    ecg, _ = run_for(build(heart_rate=88, cardiac_rhythm="STEMI"), 30.0)
    rr = rr_intervals(ecg)
    assert abs(60.0 / rr.mean() - 88) < 3
    assert rr.std() / rr.mean() < 0.06, "STEMI should stay regular"


# --- Ventricular tachycardia ------------------------------------------------
def test_vtach_overrides_the_heart_rate_slider():
    """V-Tach is driven by an ectopic ventricular focus, not the sinus node."""
    for bpm in (30, 72, 200):
        ecg, _ = run_for(build(heart_rate=bpm, cardiac_rhythm="V-Tach"), 20.0)
        measured = 60.0 / rr_intervals(ecg).mean()
        assert abs(measured - 160.0) < 3, f"slider {bpm} gave {measured:.1f} bpm"


def test_vtach_qrs_is_broad_with_no_p_or_t():
    ecg, _ = run_for(build(heart_rate=100, cardiac_rhythm="V-Tach"), 20.0)
    sinus, _ = run_for(build(heart_rate=100), 20.0)

    peak = r_peaks(ecg)[4]
    base = float(np.median(ecg[peak - 150: peak - 120]))

    wide = fwhm(ecg, peak)
    narrow = fwhm(sinus, r_peaks(sinus)[4])
    assert wide > 0.08, f"QRS is not broad ({wide * 1000:.0f} ms)"
    assert wide > 3 * narrow, f"{wide * 1000:.0f} ms vs sinus {narrow * 1000:.0f} ms"

    assert ecg[peak: peak + 200].min() - base < -0.4, "no deep S deflection"
    # Nothing repolarises: no T wave between this beat and the next.
    assert ecg[peak + 120: peak + 250].max() - base < 0.20, "a T wave is present"


def test_vtach_is_monomorphic_and_regular():
    ecg, _ = run_for(build(cardiac_rhythm="V-Tach"), 20.0)
    rr = rr_intervals(ecg)
    assert rr.std() < 0.005, f"V-Tach should be regular (std {rr.std():.4f} s)"

    peaks = r_peaks(ecg)[2:10]
    heights = np.array([ecg[p] for p in peaks])
    assert heights.std() < 0.05, "complexes vary in height; should be monomorphic"


# --- Cheyne-Stokes respiration ----------------------------------------------
def test_cheyne_stokes_produces_a_true_apnoea():
    _, resp = run_for(build(resp_pattern="Cheyne-Stokes"), 60.0)
    apnoea = resp[32 * FS: 58 * FS]
    assert np.all(apnoea == 0.0), "breathing continues through the apnoeic phase"

    breathing = resp[10 * FS: 20 * FS]
    assert np.abs(breathing).max() > 0.8, "no breathing during the ventilatory phase"


def test_cheyne_stokes_waxes_and_wanes():
    """Crescendo-decrescendo: depth peaks mid-cycle, not at its edges."""
    _, resp = run_for(build(resp_pattern="Cheyne-Stokes"), 60.0)
    depth = [np.abs(resp[s * FS: (s + 3) * FS]).max() for s in range(0, 30, 3)]
    peak_at = int(np.argmax(depth)) * 3
    assert 9 <= peak_at <= 18, f"envelope peaks at {peak_at}s, expected mid-cycle"
    assert depth[0] < 0.5 * max(depth), "no crescendo at the start of the cycle"
    assert depth[-1] < 0.6 * max(depth), "no decrescendo at the end of the cycle"


def test_normal_breathing_never_pauses():
    _, resp = run_for(build(resp_pattern="Normal"), 60.0)
    for start in range(0, 55, 5):
        window = resp[start * FS: (start + 5) * FS]
        assert np.abs(window).max() > 0.5, f"unexpected pause at {start}s"


# --- switching and interaction ----------------------------------------------
def test_rhythm_can_be_switched_live():
    engine = build(heart_rate=72)
    run_for(engine, 6.0)
    engine.state.update(cardiac_rhythm="V-Tach")
    run_for(engine, 1.0)                       # flush the scheduling lookahead
    ecg, _ = run_for(engine, 15.0)
    assert abs(60.0 / rr_intervals(ecg).mean() - 160.0) < 5

    engine.state.update(cardiac_rhythm="Sinus")
    run_for(engine, 1.0)
    ecg, _ = run_for(engine, 15.0)
    assert abs(60.0 / rr_intervals(ecg).mean() - 72) < 4, "did not return to sinus"


# Rhythms with no organised complexes at all: an injected PVC has nothing to
# couple to and is deliberately not rendered.
NON_BEAT_RHYTHMS = ("V-Fib", "Asystole")


def test_injections_still_fire_under_every_rhythm():
    """Guard: the pathology rewrite must not swallow the one-shot triggers."""
    for rhythm in CARDIAC_RHYTHMS:
        engine = build(heart_rate=72, cardiac_rhythm=rhythm)
        run_for(engine, 4.0)

        if rhythm not in NON_BEAT_RHYTHMS:
            # Count the delta: PVC Bigeminy generates ectopics of its own.
            before = sum(1 for _, kind in engine.beat_log if kind == "pvc")
            engine.state.trigger_pvc()
            run_for(engine, 4.0)
            after = sum(1 for _, kind in engine.beat_log if kind == "pvc")
            spontaneous = 3 if rhythm == "PVC Bigeminy" else 0
            assert after > before + spontaneous - 1,                 f"{rhythm}: PVC button added no ectopic beat"
        else:
            run_for(engine, 4.0)

        engine.state.trigger_motion()
        during, _ = run_for(engine, 2.0)
        assert np.abs(during).max() > 1.5, f"{rhythm}: motion artifact never fired"


def test_pathology_composes_with_the_noise_stack():
    """A pathology must not displace the artifact pipeline; they superimpose.

    Measured differentially - Gaussian noise lifts the whole spectral floor, so
    a line-to-neighbour ratio would understate a mains hum that is in fact intact.
    """
    common = dict(heart_rate=72, cardiac_rhythm="AFib", gaussian_sigma=0.05)
    quiet, _ = run_for(build(**common), 8.0)
    hum, _ = run_for(build(mains_amplitude=0.2, **common), 8.0)
    assert band_power(hum, 49.5, 50.5) > 20 * band_power(quiet, 49.5, 50.5)

    # ...and the rhythm's own signature survives the artifact stack.
    assert rr_intervals(hum).std() / rr_intervals(hum).mean() > 0.12


# --- state validation -------------------------------------------------------
def test_state_accepts_every_advertised_option():
    state = SimulationState()
    for rhythm in CARDIAC_RHYTHMS:
        assert state.update(cardiac_rhythm=rhythm).cardiac_rhythm == rhythm
    for pattern in RESP_PATTERNS:
        assert state.update(resp_pattern=pattern).resp_pattern == pattern


def test_state_rejects_an_unknown_rhythm():
    state = SimulationState()
    for bad in ("Vtach", "afib", "", None, 3):
        try:
            state.update(cardiac_rhythm=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")
    assert state.snapshot().cardiac_rhythm == "Sinus", "a rejected value was applied"


def test_defaults_are_healthy():
    params = SimulationState().snapshot()
    assert params.cardiac_rhythm == "Sinus"
    assert params.resp_pattern == "Normal"


# --- UI wiring --------------------------------------------------------------
def test_dropdowns_reach_the_state():
    state = SimulationState()
    window = MainWindow(state.snapshot())
    window.bind(state, RingBuffer())
    try:
        window.controls.rhythm_combo.setCurrentText("STEMI")
        assert state.snapshot().cardiac_rhythm == "STEMI"
        window.controls.resp_combo.setCurrentText("Cheyne-Stokes")
        assert state.snapshot().resp_pattern == "Cheyne-Stokes"
    finally:
        window.close()


def test_dropdowns_offer_exactly_the_supported_options():
    window = MainWindow()
    try:
        rhythms = [window.controls.rhythm_combo.itemText(i)
                   for i in range(window.controls.rhythm_combo.count())]
        patterns = [window.controls.resp_combo.itemText(i)
                    for i in range(window.controls.resp_combo.count())]
        assert tuple(rhythms) == CARDIAC_RHYTHMS
        assert tuple(patterns) == RESP_PATTERNS
    finally:
        window.close()


def test_bind_syncs_dropdowns_without_writing_back():
    state = SimulationState(cardiac_rhythm="AFib", resp_pattern="Cheyne-Stokes")
    window = MainWindow()                       # built with healthy defaults
    try:
        window.bind(state, RingBuffer())
        assert window.controls.rhythm_combo.currentText() == "AFib"
        assert window.controls.resp_combo.currentText() == "Cheyne-Stokes"
        assert state.snapshot().cardiac_rhythm == "AFib", "sync overwrote the state"
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
