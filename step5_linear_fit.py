"""Create a linear straight line based on step 4:
   delay = intercept + slope * buffer duration
Reads results/latency_results.csv, writes results/latency_fit.png.

Usage:
    python3 step5_linear_fit.py     (run step 4 first)
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FS = 44100
# old csv files have no sample_rate column


blocksizes, delays_ms, rates = [], [], []
with open("results/latency_results.csv") as f:
    for row in csv.DictReader(f):
        blocksizes.append(int(row["blocksize"]))
        delays_ms.append(float(row["delay_ms"]))
        rates.append(int(row.get("sample_rate") or DEFAULT_FS))
if len(set(rates)) > 1:
    raise SystemExit(f"the csv mixes sample rates {sorted(set(rates))}, fit them separately")
fs = rates[0]
blocksizes = np.array(blocksizes)
delays_ms = np.array(delays_ms)

buffer_ms = blocksizes / fs * 1000

slope, intercept = np.polyfit(buffer_ms, delays_ms, 1)
pred = slope * buffer_ms + intercept
r2 = 1 - np.sum((delays_ms - pred) ** 2) / np.sum((delays_ms - delays_ms.mean()) ** 2)

print(f"sample rate: {fs} Hz, {len(delays_ms)} runs")
print(f"fit: delay = {intercept:.1f} ms + {slope:.2f} x buffer duration")
print(f"R^2 = {r2:.4f}")
# the intercept is extrapolated, so also print the smallest real delay
# big buffers pull the line toward them, so fit buffers <= 1024 again on their own

print(f"smallest measured delay: {delays_ms.min():.1f} ms")
small_buffers = blocksizes <= 1024
if len(set(blocksizes[small_buffers])) >= 2:
    small_slope, small_intercept = np.polyfit(buffer_ms[small_buffers],
                                              delays_ms[small_buffers], 1)
    print(f"fit on buffers <= 1024 only: delay = {small_intercept:.1f} ms + "
          f"{small_slope:.2f} x buffer duration")

plt.figure(figsize=(8, 5))
plt.scatter(buffer_ms, delays_ms, alpha=0.5, label="measured runs")
xs = np.linspace(0, buffer_ms.max() * 1.05, 100)
plt.plot(xs, slope * xs + intercept, "r--",
         label=f"fit: {intercept:.1f} ms + {slope:.2f} x buffer")
plt.xlabel("Buffer duration (ms)")
plt.ylabel("Measured delay (ms)")
plt.title(f"Fixed delay + buffer-dependent delay ({fs} Hz, R^2 = {r2:.4f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/latency_fit.png", dpi=150)
plt.show()