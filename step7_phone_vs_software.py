"""Analyse phone recordings of the clap-response test.
Put the recordings in phone/ as rec_<buffer size>.wav, e.g. rec_256.wav.
A phone recording in m4a has to be converted first.
For every answer chirp the script finds the clap that came before it and
reports the time between the two. The result is compared with the loopback
measurement from step 4.

Usage:
    python3 step7_phone_vs_software.py
"""
import csv
import glob
import os
import re
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate, find_peaks

PHONE_DIR = "phone"
MIN_NCC = 0.15
SEARCH_BEFORE = 0.8
GUARD = 0.02
STEP6_FS = 44100

def make_template(fs):
    """The chirp that step 6 plays (called beep there), regenerated at the
    recording's sample rate. Keep the numbers in sync with step 6."""
    n = int(fs * 0.03)
    t = np.linspace(0, 0.03, n, endpoint=False)
    return chirp(t, f0=2000, f1=6000, t1=0.03, method="linear") * np.hanning(n)


def ncc_curve(x, template):
    """Normalised cross-correlation, between -1 and 1. Values near 1 mean
    the signal at that position looks like the template."""
    num = correlate(x, template, mode="valid", method="fft")
    template_len = len(template)
    # energy of x in every window of template_len samples, from a running sum
    sum_sq = np.cumsum(np.concatenate([[0.0], x ** 2]))
    energy = sum_sq[template_len:] - sum_sq[:-template_len]
    return num / (np.sqrt(np.maximum(energy, 0)) * np.sqrt(np.sum(template ** 2)) + 1e-12)


def analyze(path):
    """Find the clap/chirp pairs in one recording. Returns (samples, sample
    rate, pairs, number of chirps without a clap, detection threshold), each
    pair is (clap time s, chirp time s, delay ms, similarity)."""
    x, fs = sf.read(path)
    if x.ndim > 1:      # stereo: average the channels
        x = x.mean(axis=1)
    # abs() so an inverted chirp also counts, the similarity is then 0 to 1
    ncc = np.abs(ncc_curve(x, make_template(fs)))
    height = max(MIN_NCC, 0.5 * ncc.max())
    chirp_starts, _ = find_peaks(ncc, height=height, distance=int(0.5 * fs))

    smooth_len = max(1, int(0.001 * fs))
    envelope = np.convolve(np.abs(x), np.ones(smooth_len) / smooth_len, mode="same")
    noise = np.median(envelope)

    pairs = []
    no_clap = 0
    for chirp_start in chirp_starts:
        t_beep = chirp_start / fs
        lo = max(0, int((t_beep - SEARCH_BEFORE) * fs))
        hi = int((t_beep - GUARD) * fs)
        if hi <= lo:
            no_clap += 1
            continue
        seg = envelope[lo:hi]
        loudest = int(np.argmax(seg))
        if seg[loudest] < 10 * noise:
            no_clap += 1
            continue
        raw = np.abs(x[lo:hi])
        start = max(0, loudest - int(0.08 * fs))
        end = min(len(raw), loudest + int(0.005 * fs))
        around_clap = raw[start:end]
        threshold = max(0.1 * around_clap.max(), 10 * noise)
        above = np.nonzero(around_clap > threshold)[0]
        first = int(above[0]) if len(above) else loudest - start
        t_clap = (lo + start + first) / fs
        pairs.append((t_clap, t_beep, (t_beep - t_clap) * 1000, float(ncc[chirp_start])))
    return x, fs, pairs, no_clap, height


def software_stats(blocksize):
    try:
        with open("results/latency_results.csv") as f:
            delays_ms = [float(r["delay_ms"]) for r in csv.DictReader(f)
                         if int(r["blocksize"]) == blocksize]
    except FileNotFoundError:
        return None, None
    return (float(np.mean(delays_ms)), float(np.std(delays_ms))) if delays_ms else (None, None)


files = sorted(glob.glob(os.path.join(PHONE_DIR, "rec_*.wav")),
               key=lambda p: int(re.findall(r"rec_(\d+)", p)[0]))
if not files:
    raise SystemExit("No rec_<number>.wav files found in phone/.")

rows, results, example = [], [], None
for path in files:
    blocksize = int(re.findall(r"rec_(\d+)", path)[0])
    x, fs, pairs, no_clap, height = analyze(path)
    print(f"\n===== {os.path.basename(path)}  ({fs} Hz, {len(pairs)} pairs found) =====")
    print(f"chirp threshold {height:.2f}, {no_clap} chirp(s) skipped because no clap was found")
    for j, (t_clap, t_beep, delay_ms, similarity) in enumerate(pairs, 1):
        print(f"pair {j:2d}: clap {t_clap:7.3f} s -> chirp {t_beep:7.3f} s : {delay_ms:7.2f} ms (similarity {similarity:.2f})")
        rows.append([os.path.basename(path), blocksize, j, round(t_clap, 4), round(t_beep, 4),
                     round(delay_ms, 2), round(similarity, 3)])
    if not pairs:
        print("no clap/chirp pairs found in this recording")
        continue
    delays = np.array([pair[2] for pair in pairs])
    half_buffer_ms = blocksize / STEP6_FS * 1000 / 2
    corrected = delays.mean() + half_buffer_ms
    step4_mean, step4_std = software_stats(blocksize)
    print(f"phone: {delays.mean():.2f} +/- {delays.std():.2f} ms "
          f"(min {delays.min():.2f}, max {delays.max():.2f}, spread {np.ptp(delays):.2f}, n={len(delays)})")
    print(f"one buffer = {2 * half_buffer_ms:.2f} ms, "
          f"phone + half a buffer (corrected): {corrected:.2f} ms")
    if step4_mean is not None:
        print(f"software loopback (step 4): {step4_mean:.2f} +/- {step4_std:.2f} ms ; "
              f"difference {delays.mean() - step4_mean:+.2f} ms, "
              f"corrected {corrected - step4_mean:+.2f} ms")
    results.append((blocksize, delays, step4_mean, step4_std, corrected))
    if example is None:
        example = (x, fs, pairs[0], blocksize)     # first pair of the first file

if not results:
    raise SystemExit("No valid results.")

os.makedirs("results", exist_ok=True)
with open("results/phone_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "blocksize", "pair", "t_clap_s", "t_beep_s", "delay_ms", "similarity"])
    w.writerows(rows)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
x, fs, (t_clap, t_beep, delay_ms, similarity), blocksize = example
# show from 50 ms before the clap to 100 ms after the chirp start
i_from, i_to = int(max(0, t_clap - 0.05) * fs), int((t_beep + 0.1) * fs)
t_axis = np.arange(i_from, min(i_to, len(x))) / fs
axes[0].plot(t_axis, x[i_from:i_to], lw=0.7)
axes[0].axvline(t_clap, color="red", ls="--", label="clap onset")
axes[0].axvline(t_beep, color="green", ls="--", label="chirp start")
axes[0].set_title(f"Example pair from the phone recording (buffer {blocksize}): {delay_ms:.1f} ms")
axes[0].set_xlabel("Time (s)")
axes[0].legend()

for i, (blocksize, delays, step4_mean, step4_std, corrected) in enumerate(results):
    axes[1].scatter([i - 0.12] * len(delays), delays, alpha=0.4, color="tab:blue")
    axes[1].errorbar(i - 0.12, delays.mean(), yerr=delays.std(), fmt="o", color="tab:blue",
                     capsize=4, label="phone: clap -> chirp" if i == 0 else None)
    axes[1].scatter(i - 0.12, corrected, marker="o", s=70, facecolors="none",
                    edgecolors="tab:blue", linewidths=1.5,
                    label="phone + half a buffer" if i == 0 else None)
    if step4_mean is not None:
        axes[1].errorbar(i + 0.12, step4_mean, yerr=step4_std, fmt="D", color="tab:red", capsize=4,
                         label="software loopback (step 4)" if i == 0 else None)
axes[1].set_xticks(range(len(results)))
axes[1].set_xticklabels([str(r[0]) for r in results])
axes[1].set_xlabel("Buffer size (samples)")
axes[1].set_ylabel("Delay (ms)")
axes[1].set_title("Phone measurement vs software measurement")
axes[1].grid(alpha=0.3)
axes[1].legend()
plt.tight_layout()
plt.savefig("results/phone_vs_software.png", dpi=150)
print("\nsaved results/phone_results.csv and results/phone_vs_software.png")
plt.show()