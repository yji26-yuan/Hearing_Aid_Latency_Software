"""Helpers shared by the measurement scripts (step 4, step 8, step 10, step 11).

step 1 and 2 don't use this file, they show the method line by line.
"""
import numpy as np
import sounddevice as sd
from scipy.signal import chirp, correlate, correlation_lags


def make_chirp(fs, duration, f0=200, f1=8000, gain=0.5):
    """Linear chirp from f0 to f1 Hz, 'duration' seconds long."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return gain * chirp(t, f0=f0, f1=f1, t1=duration, method="linear")


def find_delay(recorded, reference, fs):
    """Find where reference starts inside recorded with cross-correlation."""
    reference = reference - np.mean(reference)      # remove the DC offset first
    recorded = recorded - np.mean(recorded)
    corr = np.abs(correlate(recorded, reference, mode="full", method="fft"))
    lags = correlation_lags(len(recorded), len(reference), mode="full")
    in_range = (lags >= 0) & (lags <= len(recorded) - len(reference))
    best = int(np.argmax(np.where(in_range, corr, 0)))
    delay_samples = int(lags[best])
    ratio = float(corr[best] / (np.median(corr) + 1e-12))
    return delay_samples, delay_samples / fs * 1000, ratio


def output_rate():
    """Sample rate of the default output device."""
    return int(sd.query_devices(kind="output")["default_samplerate"])


def print_devices():
    output_device = sd.query_devices(kind="output")
    input_device = sd.query_devices(kind="input")
    print(f"output device: {output_device['name']} ({output_device['default_samplerate']:g} Hz)")
    print(f"input device:  {input_device['name']} ({input_device['default_samplerate']:g} Hz)")
