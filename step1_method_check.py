"""
Check the delay estimator on a synthetic signal.
If the estimator works, cross-correlation should give back the same delay.

Usage:
    python3 step1_method_check.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate, correlation_lags

fs = 44100          # hz
duration = 0.5      # s
true_delay = 500    # samples

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
reference = chirp(t, f0=200, f1=8000, t1=duration, method="linear")

rng = np.random.default_rng(0)
delayed = np.concatenate([np.zeros(true_delay), reference])
delayed = delayed + 0.05 * rng.standard_normal(len(delayed))

# the lag where the correlation peaks is the delay
corr = correlate(delayed, reference, mode="full")
lags = correlation_lags(len(delayed), len(reference), mode="full")
estimated_delay = lags[np.argmax(corr)]
estimated_delay_ms = estimated_delay / fs * 1000

print(f"true delay:      {true_delay} samples")
print(f"estimated delay: {estimated_delay} samples = {estimated_delay_ms:.3f} ms")

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

n_samples_to_plot = 2000    # plot the first 2000 samples
axes[0].plot(reference[:n_samples_to_plot], label="reference")
axes[0].plot(delayed[:n_samples_to_plot], label="delayed", alpha=0.7)
axes[0].set_title("Waveforms (first 2000 samples)")
axes[0].set_xlabel("Sample")
axes[0].legend()

near_zero = (lags > -2000) & (lags < 2000)    # zoom in
axes[1].plot(lags[near_zero], corr[near_zero])
axes[1].axvline(estimated_delay, color="red", linestyle="--", label=f"peak = {estimated_delay}")
axes[1].set_title("Cross-correlation")
axes[1].set_xlabel("Lag (samples)")
axes[1].legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/fig1_synthetic_validation.png", dpi=150)
plt.show()