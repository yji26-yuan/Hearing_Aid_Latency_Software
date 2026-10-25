"""How stable is the delay within one stream, and from one stream to the next?

For every sample rate / buffer pair in CONFIGS the duplex stream is opened OPENS times.

Usage:
    python3 step10_stream_stability.py          (measure, about 90 s of chirps)
    python3 step10_stream_stability.py plot     (only redraw from the saved csv)

Writes results/stream_stability.csv and results/stream_stability.png.
"""
import csv
import os
import sys
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from common import make_chirp, find_delay, print_devices

CONFIGS = [(44100, 256), (44100, 1024), (48000, 256), (48000, 1024)]
OPENS = 5           # times each stream is opened
SECONDS = 3.0       # how long each stream stays open
SLOT = 0.25         # one chirp every 0.25 s
CHIRP_LEN = 0.06    # s
MIN_RATIO = 30      # below this: no clear chirp


def run_once(fs, blocksize):
    """Open one stream, play a chirp in every slot, return (delays in ms per
    slot, peak ratios, status flags seen)."""
    slot_samples = int(fs * SLOT)
    reference = make_chirp(fs, CHIRP_LEN, f0=500, f1=8000) * np.hanning(int(fs * CHIRP_LEN))
    pattern = np.zeros(slot_samples, dtype=np.float32)
    pattern[:len(reference)] = reference
    total = int(fs * (SECONDS + 1))
    recorded = np.zeros(total, dtype=np.float32)
    state = {"pos": 0, "flags": []}

    def callback(indata, outdata, frames, time_info, status):
        pos = state["pos"]
        idx = pos + np.arange(frames)
        outdata[:, 0] = pattern[idx % slot_samples]
        if pos + frames <= total:
            recorded[pos:pos + frames] = indata[:, 0]
        if status:
            state["flags"].append(str(status))
        state["pos"] = pos + frames

    with sd.Stream(samplerate=fs, blocksize=blocksize, channels=1, dtype="float32",
                   latency="low", callback=callback):
        time.sleep(SECONDS)

    delays, ratios = [], []
    n_slots = min(state["pos"], total) // slot_samples - 1
    for k in range(n_slots):
        _, delay_ms, ratio = find_delay(
            recorded[k * slot_samples:(k + 1) * slot_samples].astype(np.float64), reference, fs)
        delays.append(delay_ms)
        ratios.append(ratio)
    return delays, ratios, state["flags"]


os.makedirs("results", exist_ok=True)
rows = []
if len(sys.argv) > 1 and sys.argv[1] == "plot":
    with open("results/stream_stability.csv") as f:
        for r in csv.DictReader(f):
            rows.append([int(r["sample_rate"]), int(r["blocksize"]), int(r["open"]),
                         int(r["chirp"]), float(r["time_s"]), float(r["delay_ms"]),
                         float(r["peak_ratio"]), int(r["flags"]), int(r["valid"])])
else:
    print_devices()
    for fs, blocksize in CONFIGS:
        print(f"\n===== {fs} Hz, buffer {blocksize} =====")
        for opening in range(1, OPENS + 1):
            delays, ratios, flags = run_once(fs, blocksize)
            good = [d for d, r in zip(delays, ratios) if r >= MIN_RATIO]
            for chirp_index, (d, r) in enumerate(zip(delays, ratios)):
                rows.append([fs, blocksize, opening, chirp_index, round(chirp_index * SLOT, 2),
                             round(d, 3), round(r, 1), int(bool(flags)), int(r >= MIN_RATIO)])
            if good:
                print(f"open {opening}: {len(good)}/{len(delays)} chirps, "
                      f"{np.median(good):7.2f} ms, "
                      f"spread inside this open {np.ptp(good):.2f} ms"
                      f"{'  flags: ' + ', '.join(sorted(set(flags))) if flags else ''}")
            else:
                print(f"open {opening}: no clear chirp, is the volume up?")
            time.sleep(0.5)

    with open("results/stream_stability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_rate", "blocksize", "open", "chirp", "time_s", "delay_ms",
                    "peak_ratio", "flags", "valid"])
        w.writerows(rows)

# summary: spread inside each open, and how far apart the opens are
print("\n===== summary =====")
print("rate    buffer  max spread inside one open   median of each open (ms)")
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for ax, (fs, blocksize) in zip(axes.flat, CONFIGS):
    inside, medians = [], []
    for opening in range(1, OPENS + 1):
        sel = [r for r in rows
               if r[0] == fs and r[1] == blocksize and r[2] == opening and r[8] == 1]
        if not sel:
            continue
        d = np.array([r[5] for r in sel])
        inside.append(np.ptp(d))
        medians.append(np.median(d))
        ax.plot([r[4] for r in sel], d, "o-", ms=3, lw=1, label=f"open {opening}")
    if medians:
        print(f"{fs:6d} {blocksize:7d}   {max(inside):8.2f}                     "
              + "  ".join(f"{m:.2f}" for m in medians))
    ax.set_title(f"{fs} Hz, buffer {blocksize}")
    ax.set_ylabel("Delay (ms)")
    ax.grid(alpha=0.3)
    ax.margins(y=0.15)
for ax in axes[1]:
    ax.set_xlabel("Time since the stream was opened (s)")
fig.suptitle("Delay of every chirp, each line is one opening of the stream")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=OPENS, frameon=False)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig("results/stream_stability.png", dpi=150)
print("\nsaved results/stream_stability.csv and results/stream_stability.png")
plt.show()
