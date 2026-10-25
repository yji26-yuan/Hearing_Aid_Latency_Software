"""Live latency monitor.

A chirp is played once per second through a duplex audio stream while the
microphone is recorded. Each second the recording is cross-correlated with
the chirp to get the current latency. The latency is counted from the
sample where a chirp is handed to the output to the sample where it shows
up in the input.

Usage:
    python3 step8_realtime_monitor.py [buffer] [sample rate] [seconds] [live plot]
  ex: python3 step8_realtime_monitor.py 256 48000 30 0 
Defaults: 256, the speaker's rate, 30 s, no live plot.

seconds = 0 runs until Ctrl+C.

A glitch marks that period and every later period invalid,
because dropped or repeated samples move the playback and the recording against each other.

Writes results/realtime_<rate>_<buffer>.csv and .png
"""
import csv
import os
import sys
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from common import make_chirp, find_delay, output_rate, print_devices

BLOCKSIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 256
FS = int(sys.argv[2]) if len(sys.argv) > 2 else output_rate()
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0     # length of the whole run in seconds
LIVE_PLOT = (int(sys.argv[4]) if len(sys.argv) > 4 else 0) == 1

PERIOD = 1.0
CHIRP_LEN = 0.25
MIN_PEAK = 0.01
MIN_RATIO = 30      # correlation peak / median below this: no clear chirp

PERIOD_SAMPLES = int(FS * PERIOD)
reference = make_chirp(FS, CHIRP_LEN)
pattern = np.zeros(PERIOD_SAMPLES, dtype=np.float32)
pattern[:len(reference)] = reference

RING_SAMPLES = 4 * PERIOD_SAMPLES
ring = np.zeros(RING_SAMPLES, dtype=np.float32)
state = {"pos": 0, "glitches": 0, "glitches_seen": 0}


def callback(indata, outdata, frames, time_info, status):
    idx = state["pos"] + np.arange(frames)      # stream sample numbers of this buffer
    outdata[:, 0] = pattern[idx % PERIOD_SAMPLES]
    ring[idx % RING_SAMPLES] = indata[:, 0]
    state["pos"] += frames
    if status and state["pos"] >= PERIOD_SAMPLES:   # flags during period 0 (start-up) are not counted
        state["glitches"] += 1


def process(k):
    """Cross-correlate the recording of period k with the chirp. Returns
    (recorded samples, delay in samples, delay in ms, recording peak,
    correlation peak / median)."""
    idx = (k * PERIOD_SAMPLES + np.arange(PERIOD_SAMPLES)) % RING_SAMPLES
    seg = ring[idx].astype(np.float64)
    peak = float(np.max(np.abs(seg)))
    delay_samples, delay_ms, ratio = find_delay(seg, reference, FS)
    return seg, delay_samples, delay_ms, peak, ratio


def draw(ax1, ax2, log, seg, delay_ms):
    """Redraw both plots. delay_ms is the delay of the latest period."""
    # log columns used here: r[1] time, r[3] delay in ms, r[8] valid
    good = [r for r in log if r[8] == 1]
    ax1.clear()
    if good:
        ax1.plot([r[1] for r in good], [r[3] for r in good], "o-", ms=4)
        recent = [r[3] for r in good[-10:]]
        ax1.set_title(f"Latency: {good[-1][3]:.1f} ms | last {len(recent)}: "
                      f"{np.mean(recent):.1f} +/- {np.std(recent):.1f} ms | "
                      f"buffer {BLOCKSIZE}, {FS} Hz")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(alpha=0.3)
    ax2.clear()
    step = max(1, len(seg) // 2000)
    n_used = len(seg) // step * step
    envelope = np.abs(seg[:n_used]).reshape(-1, step).max(axis=1)
    ax2.plot(np.arange(len(envelope)) * step / FS * 1000, envelope, lw=0.8)
    ax2.axvline(delay_ms, color="red", ls="--", label=f"detected arrival: {delay_ms:.1f} ms")
    ax2.set_xlabel("Time since the chirp started (ms)")
    ax2.set_title("Latest recorded window (envelope)")
    ax2.legend()


print(f"buffer {BLOCKSIZE}, sample rate {FS} Hz, {DURATION:g} s, "
      f"live plot {'on' if LIVE_PLOT else 'off'}. Ctrl+C stops early.")
print_devices()

os.makedirs("results", exist_ok=True)

log = []
last = None     # samples, delay in ms of the latest period, for the final plot
fig = ax1 = ax2 = None
if LIVE_PLOT:
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.45)

try:
    stream = sd.Stream(samplerate=FS, blocksize=BLOCKSIZE, channels=1,
                       dtype="float32", latency="low", callback=callback)
except Exception as e:
    raise SystemExit(f"could not open the audio device at {FS} Hz, buffer {BLOCKSIZE}: {e}")

with stream:
    # PortAudio's own estimate (input, output) in seconds, for comparison
    reported_ms = (stream.latency[0] + stream.latency[1]) * 1000
    print(f"latency reported by the driver: in {stream.latency[0] * 1000:.1f} ms + "
          f"out {stream.latency[1] * 1000:.1f} ms = {reported_ms:.1f} ms")
    start_time = time.time()
    k = 1
    try:
        while DURATION == 0 or time.time() - start_time < DURATION:
            if state["pos"] >= (k + 1) * PERIOD_SAMPLES:
                stale = state["pos"] - (k + 1) * PERIOD_SAMPLES > RING_SAMPLES - PERIOD_SAMPLES
                seg, delay_samples, delay_ms, peak, ratio = process(k)
                new_glitches = state["glitches"] - state["glitches_seen"]
                state["glitches_seen"] = state["glitches"]
                ok = (peak >= MIN_PEAK and ratio >= MIN_RATIO
                      and state["glitches"] == 0 and not stale)
                log.append([k, round(k * PERIOD, 2), delay_samples, round(delay_ms, 3),
                            round(peak, 4), round(ratio, 1), new_glitches, int(stale), int(ok)])
                last = (seg, delay_ms)
                print(f"period {k:3d}: {delay_ms:8.3f} ms  peak={peak:.3f}  ratio={ratio:7.1f}  "
                      f"glitches={new_glitches}{'  stale' if stale else ''}  "
                      f"{'OK' if ok else 'invalid'}")
                if LIVE_PLOT:
                    draw(ax1, ax2, log, seg, delay_ms)
                    plt.pause(0.01)
                k += 1
            elif LIVE_PLOT:
                plt.pause(0.02)
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass

if not log:
    raise SystemExit("stopped before the first period was analysed, nothing saved")

valid_ms = np.array([r[3] for r in log if r[8] == 1])       # delays of the valid periods
tag = f"{FS}_{BLOCKSIZE}"
with open(f"results/realtime_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["period", "time_s", "delay_samples", "delay_ms", "peak_amp",
                "peak_ratio", "glitches", "stale", "valid"])
    w.writerows(log)
if not LIVE_PLOT:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.45)
    draw(ax1, ax2, log, last[0], last[1])
fig.savefig(f"results/realtime_{tag}.png", dpi=150)

print("\n===== summary =====")
print(f"buffer {BLOCKSIZE}, {FS} Hz: {len(log)} periods, {len(valid_ms)} valid, "
      f"{state['glitches']} audio glitches flagged")
if len(valid_ms):
    print(f"latency: {valid_ms.mean():.2f} +/- {valid_ms.std():.2f} ms "
          f"(min {valid_ms.min():.2f}, max {valid_ms.max():.2f}); driver reported {reported_ms:.1f} ms")
if state["glitches"] > 0:
    print("note: the audio glitched during the run. The period with the first glitch "
          "and all later ones are marked invalid.")
print(f"saved results/realtime_{tag}.csv and results/realtime_{tag}.png")
