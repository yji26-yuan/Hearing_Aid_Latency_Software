"""Compare the live-monitor results at 44.1 kHz and 48 kHz.

Reads results/realtime_<rate>_<buffer>.csv and writes
results/samplerate_comparison.png.

The bars show the median and the error bars the middle half of the periods
(25th to 75th percentile).

The rate is the one the stream was opened with. The same buffer size is a
slightly different duration at the two rates. On the MacBook used here the
speakers stay at 44.1 kHz, so a 48 kHz stream is resampled by the system.

Usage:
    python3 step9_samplerate_compare.py   (run step 8 first for each rate and buffer)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

FS_LIST = [44100, 48000]
BLOCKSIZES = [256, 1024]
SPREAD_WARN = 5.0
JUMP_MS = 1.0


def load(fs, blocksize):
    path = f"results/realtime_{fs}_{blocksize}.csv"
    if not os.path.exists(path):
        return np.array([])
    with open(path) as f:
        return np.array([float(r["delay_ms"]) for r in csv.DictReader(f)
                         if r["valid"] == "1"])


print("rate    buffer  valid   median  [25%..75%]        mean +/- std   spread (ms)  jumps")
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.35
for j, fs in enumerate(FS_LIST):
    medians, low_err, high_err = [], [], []
    for blocksize in BLOCKSIZES:
        delays = load(fs, blocksize)
        if len(delays):
            q25, median, q75 = np.percentile(delays, [25, 50, 75])
            medians.append(median)
            low_err.append(median - q25)
            high_err.append(q75 - median)
            spread = np.ptp(delays)
            jumps = int(np.sum(np.abs(np.diff(delays)) > JUMP_MS))
            print(f"{fs:6d} {blocksize:7d} {len(delays):6d}   {median:7.2f}  "
                  f"[{q25:7.2f}..{q75:7.2f}]  "
                  f"{delays.mean():7.2f} +/- {delays.std():5.2f}   {spread:6.2f}      {jumps:3d}"
                  f"{'   <- not stable, check this run' if spread > SPREAD_WARN else ''}")
        else:
            medians.append(np.nan)
            low_err.append(0)
            high_err.append(0)
            print(f"{fs:6d} {blocksize:7d}      0   no valid data")
    # bars of the two rates side by side around each tick
    x = np.arange(len(BLOCKSIZES)) + (j - 0.5) * width
    ax.bar(x, medians, width, yerr=[low_err, high_err], capsize=4, label=f"{fs} Hz")
ax.set_xticks(np.arange(len(BLOCKSIZES)))
ax.set_xticklabels([str(b) for b in BLOCKSIZES])
ax.set_xlabel("Buffer size (samples)")
ax.set_ylabel("Measured latency (ms)")
ax.set_title("Latency at 44.1 kHz vs 48 kHz (median, 25th-75th percentile)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/samplerate_comparison.png", dpi=150)
print("\nsaved results/samplerate_comparison.png")
plt.show()
