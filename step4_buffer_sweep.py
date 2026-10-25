"""Loopback measurement for several buffer sizes, several runs each.
Writes results/latency_results.csv and results/latency_vs_blocksize.png.

Usage:
    python3 step4_buffer_sweep.py    
"""
import os
import csv
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from common import make_chirp, find_delay, output_rate, print_devices

BLOCKSIZES = [64, 128, 256, 512, 1024, 2048, 4096]
REPEATS = 5         # runs per buffer size
MAX_TRIES = 3       # tries before giving up on a run
MIN_PEAK = 0.01     # below this means too quiet
MIN_RATIO = 30      # below this means too noisy
os.makedirs("results", exist_ok=True)

fs = output_rate()
print_devices()
print("sample rate:", fs, "Hz")

reference = make_chirp(fs, 0.5)
playback = np.concatenate([reference, np.zeros(int(fs * 1.0))])  # leave 1 second of silence at the end.


def measure_once(blocksize):
    recorded = sd.playrec(playback, samplerate=fs, channels=1,
                          blocksize=blocksize, latency="low")
    sd.wait()
    glitched = bool(sd.get_status())
    recorded = recorded[:, 0]
    peak = float(np.max(np.abs(recorded)))
    delay_samples, delay_ms, ratio = find_delay(recorded, reference, fs)
    return delay_samples, delay_ms, peak, ratio, glitched


rows = []
for blocksize in BLOCKSIZES:
    for run in range(1, REPEATS + 1):
        for attempt in range(1, MAX_TRIES + 1):
            delay_samples, delay_ms, peak, ratio, glitched = measure_once(blocksize)
            ok = (peak >= MIN_PEAK) and (ratio >= MIN_RATIO) and not glitched
            if ok:
                verdict = "OK"
            elif attempt < MAX_TRIES:
                verdict = "rejected, retrying"
            else:
                verdict = "rejected, giving up on this run"
            print(f"blocksize={blocksize:5d} run {run} (try {attempt}): {delay_samples:6d} samples = "
                  f"{delay_ms:8.3f} ms  peak={peak:.3f}  ratio={ratio:7.1f}  "
                  f"{'glitch  ' if glitched else ''}{verdict}")
            time.sleep(0.5)     # short pause before the next stream is opened
            if ok:
                rows.append([fs, blocksize, run, delay_samples, round(delay_ms, 3),
                             round(peak, 4), round(ratio, 1)])
                break

if not rows:
    raise SystemExit("No valid runs. Raise the volume and keep the room quiet.")

with open("results/latency_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_rate", "blocksize", "run", "delay_samples", "delay_ms",
                "peak_amp", "peak_ratio"])
    w.writerows(rows)

print("\n===== summary (mean +/- std, min~max) =====")
# row columns: 0 sample_rate, 1 blocksize, 2 run, 3 delay_samples, 4 delay_ms, 5 peak_amp, 6 peak_ratio
measured_blocksizes, means, stds = [], [], []
for blocksize in BLOCKSIZES:
    delays_ms = [r[4] for r in rows if r[1] == blocksize]
    if not delays_ms:
        print(f"blocksize={blocksize:5d}: no valid runs")
        continue
    measured_blocksizes.append(blocksize)
    means.append(np.mean(delays_ms))
    stds.append(np.std(delays_ms))
    print(f"blocksize={blocksize:5d}: {np.mean(delays_ms):7.2f} +/- {np.std(delays_ms):5.2f} ms  "
          f"({min(delays_ms):.2f}~{max(delays_ms):.2f}, n={len(delays_ms)})")

plt.figure(figsize=(9, 5))
plt.scatter([r[1] for r in rows], [r[4] for r in rows],
            alpha=0.4, label="individual runs")
plt.errorbar(measured_blocksizes, means, yerr=stds, fmt="o-", color="red",
             capsize=4, label="mean +/- std")
plt.xscale("log", base=2)
plt.xticks(BLOCKSIZES, [str(b) for b in BLOCKSIZES])
plt.xlabel("Buffer size (samples)")
plt.ylabel("Measured delay (ms)")
plt.title(f"Speaker-to-microphone delay vs buffer size ({fs} Hz)\n"
          f"{sd.query_devices(kind='output')['name']} -> {sd.query_devices(kind='input')['name']}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/latency_vs_blocksize.png", dpi=150)
print("\nsaved results/latency_results.csv and results/latency_vs_blocksize.png")
plt.show()
