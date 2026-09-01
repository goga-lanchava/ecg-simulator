"""Step 1 checks: state clamping / one-shot flags, and ring-buffer wrap-around.

Run directly (``python tests/test_core.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.buffer import RingBuffer                      # noqa: E402
from core.state import BUFFER_SIZE, CHUNK_SAMPLES, SimulationState  # noqa: E402


def test_constants():
    assert BUFFER_SIZE == 10_000       # 10 s at 1 kHz
    assert CHUNK_SAMPLES == 50         # 50 ms per producer tick


def test_state_defaults_and_clamping():
    s = SimulationState()
    assert s.snapshot().heart_rate == 72.0

    s.update(heart_rate=500, respiratory_rate=1, gaussian_sigma=-1)
    p = s.snapshot()
    assert p.heart_rate == 200.0        # clamped to HR_RANGE high
    assert p.respiratory_rate == 8.0    # clamped to RR_RANGE low
    assert p.gaussian_sigma == 0.0      # clamped to GAUSSIAN_RANGE low

    try:
        s.update(hart_rate=80)
    except KeyError:
        pass
    else:
        raise AssertionError("misspelled parameter should raise")


def test_snapshot_is_immutable_and_detached():
    s = SimulationState()
    snap = s.snapshot()
    s.update(heart_rate=150)
    assert snap.heart_rate == 72.0      # the producer's copy did not move
    assert s.snapshot().heart_rate == 150.0


def test_one_shot_flags():
    s = SimulationState()
    assert s.consume_motion() is False
    s.trigger_motion()
    assert s.consume_motion() is True
    assert s.consume_motion() is False  # a single click fires exactly once

    s.trigger_pvc()
    s.trigger_pvc()                     # coalesces; still one event
    assert s.consume_pvc() is True
    assert s.consume_pvc() is False


def test_flag_consumed_exactly_once_under_contention():
    s = SimulationState()
    s.trigger_motion()
    hits = []

    def worker():
        if s.consume_motion():
            hits.append(1)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(hits) == 1


def test_ring_buffer_wraps_and_stays_aligned():
    rb = RingBuffer(capacity=100, channels=("ecg", "resp"), sample_rate=10)
    assert rb.duration == 10.0

    for k in range(8):                  # 8 x 30 = 240 samples over a 100-slot buffer
        base = k * 30
        rb.write({
            "ecg": np.arange(base, base + 30, dtype=float),
            "resp": -np.arange(base, base + 30, dtype=float),
        })

    assert rb.write_index == 240 % 100
    assert rb.total_written == 240

    chron = rb.chronological()
    assert chron.shape == (2, 100)
    assert chron[0, 0] == 140 and chron[0, -1] == 239      # oldest -> newest
    assert np.allclose(chron[1], -chron[0])                # channels stay aligned
    assert np.array_equal(rb.latest(5)[0], np.arange(235.0, 240.0))


def test_snapshot_returns_a_copy():
    rb = RingBuffer(capacity=16, channels=("a",))
    rb.write(np.arange(16.0))
    data, index = rb.snapshot()
    assert index == 0                    # exactly one lap
    data[0, 0] = 999.0
    assert rb.snapshot()[0][0, 0] == 0.0


def test_partial_fill_and_oversized_chunk():
    rb = RingBuffer(capacity=10, channels=("a",))
    rb.write(np.arange(4.0))
    assert np.array_equal(rb.chronological()[0], np.arange(4.0))  # no NaN padding

    rb.clear()
    assert rb.total_written == 0
    rb.write(np.arange(25.0))            # longer than the buffer: keep the tail
    assert np.array_equal(rb.chronological()[0], np.arange(15.0, 25.0))


def test_write_validation():
    rb = RingBuffer(capacity=10, channels=("ecg", "resp"))
    for bad, exc in (
        ({"ecg": np.zeros(5)}, KeyError),                                  # missing channel
        ({"ecg": np.zeros(5), "resp": np.zeros(4)}, ValueError),           # ragged
        (np.zeros((3, 5)), ValueError),                                    # wrong channel count
    ):
        try:
            rb.write(bad)
        except exc:
            continue
        raise AssertionError(f"expected {exc.__name__} for {type(bad).__name__} input")


def test_csv_export():
    rb = RingBuffer(capacity=8, channels=("ecg", "resp"), sample_rate=4)
    rb.write({"ecg": np.arange(8.0), "resp": np.arange(8.0) * 2})
    path = os.path.join(tempfile.gettempdir(), "ringbuffer_export.csv")
    rows = rb.to_csv(path)
    assert rows == 8

    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "time_s,ecg,resp"
    assert lines[1].startswith("0.000000,0.000000,0.000000")
    assert lines[-1].startswith("1.750000,7.000000,14.000000")   # 7 / 4 Hz
    os.remove(path)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)} passed")
