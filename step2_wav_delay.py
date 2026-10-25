"""Measure the delay between a reference WAV file and a recorded WAV file.

Usage:
    python3 step2_wav_delay.py reference.wav recorded.wav 
    python3 step2_wav_delay.py 
"""
import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate, correlation_lags


def load_mono(path):
    data, fs = sf.read(path)
    if data.ndim > 1:       # one left channel besides stereo
        data = data[:, 0]
    return data, fs


def measure_latency(ref_path, rec_path, save_as="results/fig2_wav_measurement.png"):
    """Cross-correlate the two files, plot the result and save the figure.
    Returns (delay in samples, delay in ms)."""
    reference, fs_ref = load_mono(ref_path)
    recorded, fs_rec = load_mono(rec_path)
    if fs_ref != fs_rec:
        raise ValueError(f"sample rates differ: {fs_ref} vs {fs_rec}")
    fs = fs_ref

    # remove the DC offset before correlating
    reference = reference - np.mean(reference)
    recorded = recorded - np.mean(recorded)

    corr = correlate(recorded, reference, mode="full")
    lags = correlation_lags(len(recorded), len(reference), mode="full")
    # abs() can find the negative peaks.
    delay_samples = lags[np.argmax(np.abs(corr))]
    delay_ms = delay_samples / fs * 1000

    print(f"sample rate: {fs} Hz")
    print(f"delay: {delay_samples} samples = {delay_ms:.3f} ms")
    if delay_samples < 0:
        print("note!!! negative delay, Are the two file names swapped?")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    t_ref_ms = np.arange(len(reference)) / fs * 1000
    t_rec_ms = np.arange(len(recorded)) / fs * 1000
    axes[0].plot(t_ref_ms, reference, label="reference")
    axes[0].plot(t_rec_ms, recorded, label="recorded", alpha=0.7)
    axes[0].set_xlim(0, max(100, delay_ms + 50))
    axes[0].set_title("Waveforms")
    axes[0].set_xlabel("Time (ms)")
    axes[0].legend()

    lag_ms = lags / fs * 1000
    axes[1].plot(lag_ms, corr)      # signed correlation, the peak can point down
    axes[1].axvline(delay_ms, color="red", linestyle="--",
                    label=f"delay = {delay_ms:.3f} ms")
    axes[1].set_xlim(delay_ms - 50, delay_ms + 50)
    axes[1].set_title("Cross-correlation")
    axes[1].set_xlabel("Lag (ms)")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_as) or ".", exist_ok=True)
    plt.savefig(save_as, dpi=150)
    plt.show()
    return delay_samples, delay_ms


if __name__ == "__main__":
    if len(sys.argv) == 3:
        measure_latency(sys.argv[1], sys.argv[2])
    else:
        # no files given: build a synthetic pair and measure it
        os.makedirs("wav", exist_ok=True)
        fs, true_delay, duration = 44100, 500, 0.5
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        reference = 0.5 * chirp(t, f0=200, f1=8000, t1=duration, method="linear")
        rng = np.random.default_rng(0)
        recorded = np.concatenate([np.zeros(true_delay), reference])
        recorded = recorded + 0.01 * rng.standard_normal(len(recorded))
        sf.write("wav/reference.wav", reference, fs)
        sf.write("wav/test_recorded.wav", recorded, fs)
        print(f"wrote wav/reference.wav and wav/test_recorded.wav "
              f"(true delay {true_delay} samples)")
        measure_latency("wav/reference.wav", "wav/test_recorded.wav")