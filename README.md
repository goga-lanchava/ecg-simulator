# ECG Simulator

A real-time synthetic physiological waveform generator and clinical monitor UI.
Generates mathematically modelled ECG and respiration, injects artifacts and
arrhythmias on demand, and renders them on a sweeping hospital-style monitor.

![The monitor](docs/panel_open.png)

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Python 3.10+. Verified on Python 3.14.7 with PyQt6 6.11, pyqtgraph 0.14,
numpy 2.5 and scipy 1.18.

## What it does

**Waveforms.** Each heartbeat is a sum of five Gaussians (a simplified McSharry
model) in the order P, Q, R, S, T:

```
z(t) = Σ aᵢ · exp( -(t - θᵢ)² / (2bᵢ²) )
```

Respiration is a phase-warped asymmetric wave with a controllable I:E ratio,
coupled back into the ECG two ways: as respiratory sinus arrhythmia modulating
the R-R interval, and as a slow baseline sway.

**19 cardiac rhythms** and **6 breathing patterns**, all switchable live:

| | |
|---|---|
| *Atrial* | AFib, Atrial Flutter, SVT, Junctional |
| *Blocks* | 1st Degree, Mobitz I (Wenckebach), Mobitz II, 3rd Degree |
| *Ventricular* | PVC Bigeminy, V-Tach, Torsades, V-Fib, Idioventricular, Asystole |
| *Ischaemic / metabolic* | STEMI, Ischemia (ST depression), Hyperkalemia |
| *Device* | Paced |
| *Breathing* | Normal, Cheyne-Stokes, Kussmaul, Biot (Ataxic), Apnoea, Agonal |

![Rhythm catalogue](docs/catalogue_rhythms.png)
![Breathing patterns](docs/catalogue_breathing.png)

**Artifacts.** 50 Hz mains hum, 0.5 Hz baseline wander, Gaussian sensor noise,
and an injectable motion transient — each on its own control.

**Electrical therapy.** Defibrillation (unsynchronised) and synchronised
cardioversion, with outcomes that respect the distinctions that matter:

- asystole is never shockable, however many times you press the button;
- a synchronised shock cannot be delivered without an R wave to time it to;
- an unsynchronised shock into an organised rhythm can land on the T wave and
  induce VF.

Shocks are probabilistic rather than guaranteed, and the outcome is reported
explicitly, so a failed shock reads as a modelled outcome and not a bug.
A therapy alarm flags rhythms that call for intervention — flashing for arrest
rhythms, steady for unstable-but-perfusing ones.

![Therapy alarm](docs/therapy_alarm.png)
![After defibrillation](docs/therapy_shock.png)

The second image reads left to right as one sweep: V-Fib, the discharge
artifact, the amplifier recovering from saturation, a post-shock pause, then
sinus rhythm returning.

## Architecture

A producer/consumer split across two threads, with a ring buffer as the only
synchronisation point, so a slow repaint can never stall the sample clock:

```
SignalGenerator (QThread) --writes--> RingBuffer <--reads-- MonitorView
     50 ms chunks                  10 s @ 1 kHz          QTimer, 30 FPS
```

```
main.py              entry point; wires the three pieces together
core/state.py        thread-safe parameter state and one-shot event flags
core/buffer.py       fixed-size multi-channel ring buffer
core/pathology.py    catalogue: morphologies, rhythms, breathing, therapy
core/generator.py    the waveform engine and the producer thread
ui/main_window.py    control panel, layout, therapy and alarm
ui/monitor.py        the sweep renderer
```

The display does not scroll. The x-axis is fixed at 0–10 s and the trace stays
where it was drawn; a block of `NaN` samples is blanked just ahead of the write
cursor, so what moves is the *gap* — the erase bar of a hospital monitor.

`core/pathology.py` is a data table. Adding a rhythm means editing that file,
not the engine, and the UI dropdown and the in-app description update from it
automatically.

## Tests

143 tests across seven suites. They assert on measurements taken back out of
the generated signal — R-R intervals, FFT bins, QRS widths — rather than on the
fact that numbers came out.

```bash
python tests/test_core.py         # ring buffer, state
python tests/test_generator.py    # waveform models, producer thread
python tests/test_ui.py           # widgets, layout, panel collapse
python tests/test_sweep.py        # sweep cursor, wiring, CSV export
python tests/test_pathology.py    # AFib, STEMI, V-Tach, Cheyne-Stokes
python tests/test_rhythms.py      # the full catalogue
python tests/test_therapy.py      # defibrillation, cardioversion, alarm
```

They also run under `pytest`.

### Dev tools (not tests)

```bash
python tests/preview_waveforms.py           # signal sheet -> PNG
python tests/preview_pathology.py           # the whole catalogue -> PNG
python tests/preview_window.py out.png AFib Cheyne-Stokes   # app screenshot
```

## Keyboard

| | |
|---|---|
| `Ctrl+H` / `F9` | collapse or restore the control panel |

## Not a medical device

Every waveform here is synthetic and parameterised for teaching and software
testing. It is not derived from patient data, has not been validated against
any clinical standard, and must not be used for diagnosis, for training that
substitutes for clinical instruction, or to test equipment intended for
patient care.
