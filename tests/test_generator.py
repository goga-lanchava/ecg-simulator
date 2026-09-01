"""Step 2 checks: does the generator produce *physiologically correct* signal?

These assert on measurements taken back out of the waveform - R-R intervals,
FFT bins, QRS widths - rather than on the fact that numbers came out.

Run directly (``python tests/test_generator.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.buffer import RingBuffer                                   # noqa: E402
from core.generator import (                                          # noqa: E402
    MOTION_AMPLITUDE,
    RESP_INSPIRATORY_FRACTION,
    SignalGenerator,
    WaveformEngine,
)
from core.state import CHUNK_SAMPLES, MOTION_DURATION_S, SAMPLE_RATE, SimulationState  # noqa: E402

FS = SAMPLE_RATE


def build(seed: int = 7, **params) -> WaveformEngine:
    state = SimulationState(**params)
    return WaveformEngine(state, FS, np.random.default_rng(seed))


def run_for(engine: WaveformEngine, seconds: float, chunk: int = CHUNK_SAMPLES):
    """Drive the engine chunk by chunk, exactly as the producer thread does."""
    ecg, resp = [], []
    for _ in range(int(seconds * FS / chunk)):
        block = engine.next_chunk(chunk)
        ecg.append(block["ecg"])
        resp.append(block["resp"])
    return np.concatenate(ecg), np.concatenate(resp)


def r_peaks(ecg: np.ndarray, min_rr: float = 0.2) -> np.ndarray:
    idx, _ = find_peaks(ecg, height=0.5, distance=int(min_rr * FS))
    return idx


def pr_baseline(ecg: np.ndarray, peak: int) -> float:
    """Isoelectric level from the PR segment - between the P wave and Q onset.

    Amplitudes must be read against a *local* baseline: the respiratory sway is
    a real part of the signal, so any fixed reference drifts against it.
    """
    return float(np.median(ecg[peak - 95: peak - 78]))


def fwhm(ecg: np.ndarray, peak: int) -> float:
    """Full width at half maximum of the deflection at ``peak``, in seconds.

    A robust stand-in for QRS duration: unlike a fixed-threshold crossing it is
    not confounded by the Q and S notches cutting into the R wave's flanks.
    """
    half = ecg[peak] / 2.0
    left = peak
    while left > 0 and ecg[left] > half:
        left -= 1
    right = peak
    while right < ecg.size - 1 and ecg[right] > half:
        right += 1
    return (right - left) / FS


# --- ECG rate ---------------------------------------------------------------
def test_measured_rate_tracks_commanded_bpm():
    """The whole point of the model: R-R intervals must match the slider."""
    for bpm in (40, 72, 150, 200):
        engine = build(heart_rate=bpm)
        ecg, _ = run_for(engine, 20.0)
        peaks = r_peaks(ecg)
        assert len(peaks) > 5, f"{bpm} bpm: only {len(peaks)} R peaks detected"

        measured = 60.0 / (np.diff(peaks).mean() / FS)
        assert abs(measured - bpm) / bpm < 0.02, f"commanded {bpm}, measured {measured:.1f}"


def test_rate_change_is_followed():
    engine = build(heart_rate=60)
    run_for(engine, 6.0)
    engine.state.update(heart_rate=140)
    run_for(engine, 1.0)                       # let the scheduling lookahead flush
    ecg, _ = run_for(engine, 12.0)
    measured = 60.0 / (np.diff(r_peaks(ecg)).mean() / FS)
    assert abs(measured - 140) / 140 < 0.03, f"after change measured {measured:.1f}"


# --- ECG morphology ---------------------------------------------------------
def test_five_waves_have_the_right_shape():
    """Each Gaussian must come back out at the amplitude it was specified with."""
    engine = build(heart_rate=60)
    ecg, _ = run_for(engine, 10.0)

    measured = []
    for peak in r_peaks(ecg)[2:6]:
        base = pr_baseline(ecg, peak)

        def window(lo, hi, _p=peak):
            return ecg[_p + int(lo * FS): _p + int(hi * FS)] - base

        measured.append((
            ecg[peak] - base,                  # R
            window(-0.07, -0.03).min(),        # Q
            window(0.03, 0.07).min(),          # S
            window(-0.30, -0.13).max(),        # P
            window(0.22, 0.48).max(),          # T
        ))
        # Ordering: Q is below the baseline immediately before the R peak.
        assert window(-0.30, -0.13).max() < ecg[peak] - base
        assert ecg[peak - int(0.05 * FS)] - base < 0

    r, q, s, p, t = np.array(measured).mean(axis=0)
    assert abs(r - 1.00) < 0.03, f"R {r:.3f}"
    assert abs(q - -0.15) < 0.03, f"Q {q:.3f}"
    assert abs(s - -0.15) < 0.03, f"S {s:.3f}"
    assert abs(p - 0.15) < 0.03, f"P {p:.3f}"
    assert abs(t - 0.30) < 0.06, f"T {t:.3f}"    # widest window; sways most


def test_qt_shortens_at_high_rate():
    """Bazett scaling: the T wave migrates toward the QRS as the rate climbs."""
    def t_offset(bpm):
        engine = build(heart_rate=bpm)
        ecg, _ = run_for(engine, 14.0)
        peak = r_peaks(ecg)[3]
        lo = int(0.12 * FS)
        # Stop well short of the next R peak, or argmax finds that instead.
        hi = int(min(0.5, 0.6 * 60.0 / bpm) * FS)
        return int(np.argmax(ecg[peak + lo: peak + hi])) + lo

    slow, fast = t_offset(50), t_offset(120)
    assert fast < slow, f"T offset did not shorten: {slow} -> {fast} samples"
    # Bazett predicts sqrt(RR) scaling: sqrt(0.5/1.2) = 0.65 of the slow offset.
    assert 0.55 < fast / slow < 0.75, f"scaling off Bazett: {fast / slow:.2f}"


# --- Respiration ------------------------------------------------------------
def test_respiratory_rate_and_asymmetry():
    for brpm in (8, 15, 30):
        engine = build(respiratory_rate=brpm)
        _, resp = run_for(engine, 60.0)

        peaks, _ = find_peaks(resp, distance=int(0.5 * FS))
        measured = 60.0 / (np.diff(peaks).mean() / FS)
        assert abs(measured - brpm) / brpm < 0.02, f"commanded {brpm}, measured {measured:.1f}"

        troughs, _ = find_peaks(-resp, distance=int(0.5 * FS))
        # Inspiration = trough -> next peak; expiration = peak -> next trough.
        t0 = troughs[0]
        rise = peaks[peaks > t0][0] - t0
        fall = troughs[troughs > peaks[peaks > t0][0]][0] - peaks[peaks > t0][0]
        ratio = rise / (rise + fall)
        assert abs(ratio - RESP_INSPIRATORY_FRACTION) < 0.02, f"I:E fraction {ratio:.3f}"

    assert -1.01 <= resp.min() and resp.max() <= 1.01


def test_rsa_modulates_the_rr_interval():
    """Faster during inspiration, slower during expiration - real RSA."""
    engine = build(heart_rate=70, respiratory_rate=12)
    ecg, resp = run_for(engine, 90.0)
    peaks = r_peaks(ecg)
    rr = np.diff(peaks) / FS
    assert rr.std() > 0.005, "RR intervals are flat; RSA is not being applied"

    # Short intervals should coincide with the inspiratory (positive) phase.
    phase = resp[peaks[:-1]]
    assert np.corrcoef(phase, rr)[0, 1] < -0.5, "RR does not shorten on inspiration"


# --- Noise pipeline ---------------------------------------------------------
def band_power(x: np.ndarray, lo: float, hi: float) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / FS)
    return float(spectrum[(freqs >= lo) & (freqs <= hi)].sum())


def test_mains_lands_at_50hz():
    clean, _ = run_for(build(), 8.0)
    noisy, _ = run_for(build(mains_amplitude=0.3), 8.0)

    assert band_power(noisy, 49.5, 50.5) > 50 * band_power(clean, 49.5, 50.5)
    # and it is a line, not broadband
    assert band_power(noisy, 49.5, 50.5) > 20 * band_power(noisy, 40.0, 45.0)


def test_baseline_wander_lands_at_half_hz():
    clean, _ = run_for(build(), 20.0)
    noisy, _ = run_for(build(baseline_amplitude=0.4), 20.0)
    assert band_power(noisy, 0.4, 0.6) > 10 * band_power(clean, 0.4, 0.6)


def test_gaussian_noise_scales_with_sigma():
    quiet, _ = run_for(build(gaussian_sigma=0.01), 8.0)
    loud, _ = run_for(build(gaussian_sigma=0.15), 8.0)
    hf = lambda x: np.diff(x).std()            # noqa: E731 - high-frequency content
    assert hf(loud) > 8 * hf(quiet)


def test_clean_signal_is_actually_clean():
    """With every noise slider at zero the isoelectric line carries no jitter."""
    ecg, _ = run_for(build(heart_rate=60), 8.0)
    peak = r_peaks(ecg)[3]
    quiet = ecg[peak + 480: peak + 620]        # contiguous TP segment: after T, before the next P
    assert np.diff(quiet).std() < 1e-3, "isoelectric line is not quiet with noise at zero"
    assert np.ptp(quiet) < 0.05, "unexpected excursion between beats"


# --- Artifacts --------------------------------------------------------------
def test_motion_artifact_fires_once_and_decays():
    engine = build()
    before, _ = run_for(engine, 3.0)
    engine.state.trigger_motion()
    during, _ = run_for(engine, MOTION_DURATION_S + 0.5)
    after, _ = run_for(engine, 3.0)

    assert np.abs(during).max() > 0.5 * MOTION_AMPLITUDE, "motion artifact too small"
    assert np.abs(during).max() > 2 * np.abs(before).max()
    assert np.abs(after).max() < 1.5, "artifact did not clear (flag consumed twice?)"
    assert engine.state.inject_motion is False

    # It is a transient, not a permanent offset: the tail returns to baseline.
    assert abs(during[-1]) < 1.5


def test_pvc_is_premature_wide_and_followed_by_a_pause():
    engine = build(heart_rate=60)
    run_for(engine, 4.0)
    engine.state.trigger_pvc()
    ecg, _ = run_for(engine, 6.0)

    ectopic = [t for t, kind in engine.beat_log if kind == "pvc"]
    assert len(ectopic) == 1, f"expected exactly one PVC, scheduled {len(ectopic)}"
    pvc_t = ectopic[0]

    sinus = sorted(t for t, kind in engine.beat_log if kind == "sinus")
    before = max(t for t in sinus if t < pvc_t)
    after = min(t for t in sinus if t > pvc_t)

    rr = 1.0                                    # 60 bpm
    assert pvc_t - before < 0.9 * rr, "PVC was not premature"
    assert abs((after - before) - 2 * rr) < 0.15 * rr, "no full compensatory pause"

    # The ectopic QRS must be broad and tall.  Width is measured as FWHM, which
    # is not thrown off by the Q/S notches the way a fixed threshold is.
    peaks = r_peaks(ecg)
    times = peaks / FS + (engine.elapsed - len(ecg) / FS)

    def peak_near(target: float) -> int:
        return int(peaks[int(np.argmin(np.abs(times - target)))])

    pvc_idx = peak_near(pvc_t)
    # Reference beat chosen by time, not index: the ectopic beat is itself an
    # early peak in the record, so peaks[1] can be the PVC.
    sinus_idx = peak_near(min(t for t in sinus if t > pvc_t + 1.0))

    assert ecg[pvc_idx] > 1.4, f"ectopic beat is not high-amplitude ({ecg[pvc_idx]:.2f})"
    pvc_width = fwhm(ecg, pvc_idx)
    sinus_width = fwhm(ecg, sinus_idx)
    assert sinus_width < 0.040, f"sinus QRS unexpectedly broad ({sinus_width * 1000:.0f} ms)"
    assert pvc_width > 0.080, f"ectopic QRS is not wide ({pvc_width * 1000:.0f} ms)"
    assert pvc_width > 3 * sinus_width, "PVC no wider than a sinus beat"

    # Discordant T wave: it opposes the dominant QRS deflection.
    assert ecg[pvc_idx: pvc_idx + int(0.45 * FS)].min() < -0.3, "T wave is not discordant"


# --- Continuity -------------------------------------------------------------
def test_chunking_does_not_change_the_waveform():
    """Strongest seam test: 50-sample chunks must equal one 3000-sample call."""
    a, _ = run_for(build(heart_rate=88, respiratory_rate=17), 3.0, chunk=50)
    engine = build(heart_rate=88, respiratory_rate=17)
    b = engine.next_chunk(3000)["ecg"]
    assert np.allclose(a, b, atol=1e-9), f"max seam error {np.abs(a - b).max():.2e}"


def test_no_step_discontinuities_at_chunk_boundaries():
    ecg, resp = run_for(build(heart_rate=75), 10.0)
    jumps = np.abs(np.diff(ecg))
    at_seams = jumps[CHUNK_SAMPLES - 1::CHUNK_SAMPLES]
    assert at_seams.max() <= jumps.max(), "largest jump in the record sits on a seam"
    assert np.abs(np.diff(resp)).max() < 0.01


# --- Producer thread --------------------------------------------------------
def test_producer_thread_fills_buffer_in_real_time():
    from PyQt6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([])   # noqa: F841
    state = SimulationState(heart_rate=80)
    buffer = RingBuffer(capacity=10_000, channels=("ecg", "resp"), sample_rate=FS)
    gen = SignalGenerator(state, buffer, rng=np.random.default_rng(3))

    started = time.perf_counter()
    gen.start()
    while time.perf_counter() - started < 1.5:
        time.sleep(0.05)
    written = buffer.total_written
    elapsed = time.perf_counter() - started

    assert gen.stop() is True, "producer thread did not stop cleanly"
    assert gen.isFinished()

    rate = written / elapsed
    assert abs(rate - FS) / FS < 0.10, f"sample clock drifted: {rate:.0f} Hz vs {FS} Hz"

    data, index = buffer.snapshot()
    assert index == written % buffer.capacity
    assert np.isfinite(data[0, :written]).all(), "NaNs leaked into written samples"
    assert np.abs(data[0, :written]).max() > 0.5, "no QRS complexes in the buffer"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        started = time.perf_counter()
        try:
            fn()
            print(f"PASS  {fn.__name__}  ({time.perf_counter() - started:.2f}s)")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
