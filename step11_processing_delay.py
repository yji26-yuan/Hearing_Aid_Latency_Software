"""A filters the output with each FIR length in TAPS, one after another in the same stream.
B waits in the callback for each part of the buffer in LOADS.
Both run for every buffer in BLOCKSIZES, and each condition is opened OPENS times.

Usage:
    python3 step11_processing_delay.py          (measure, 5 openings per condition)
    python3 step11_processing_delay.py 10       (measure, 10 openings per condition)
    python3 step11_processing_delay.py test     (no sound, fake loopback, writes nothing)
    python3 step11_processing_delay.py plot     (only redraw from the saved csv)

Writes results/processing_delay.csv and results/processing_delay.png.
"""
import csv
import os
import random
import sys
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from common import make_chirp, find_delay, output_rate, print_devices

BLOCKSIZES = [256, 1024]
TAPS = [1, 65, 257, 1025, 4097]
LOADS = [0, 0.5, 0.9, 1.1]
OPENS = 5
SECONDS = {"A": 6.0, "B": 4.0}
SLOT = 0.5
CHIRP_LEN = 0.06
CUTOFF = 10000
MIN_RATIO = 30
TEST_DELAY = {256: 105.9, 1024: 158.4}

mode = sys.argv[1] if len(sys.argv) > 1 else ""
rounds = int(mode) if mode.isdigit() else OPENS
fs = 44100 if mode == "test" else output_rate()
filters = {n: firwin(n, CUTOFF, fs=fs) if n > 1 else np.array([1.0]) for n in TAPS}
reference = make_chirp(fs, CHIRP_LEN, f0=500, f1=8000) * np.hanning(int(fs * CHIRP_LEN))


def run_once(blocksize, taps_per_slot, load, seconds, fake_delay=None):
    """Open one stream, play a chirp in every slot, filtered with taps_per_slot[k]
    taps. With fake_delay there is no stream: the input is the output, fake_delay
    samples later."""
    slot_samples = blocksize * round(fs * SLOT / blocksize)
    pattern = np.zeros(slot_samples, dtype=np.float32)
    pattern[:len(reference)] = reference
    total = int(fs * seconds)
    recorded = np.zeros(total, dtype=np.float32)
    state = {"pos": 0, "slot": -1, "taps": None, "zi": None, "flags": [], "proc_times": []}

    def callback(indata, outdata, frames, time_info, status):
        start = time.perf_counter()
        pos = state["pos"]
        slot = pos // slot_samples
        if slot != state["slot"]:  # the filter tail has died out long before the slot ends
            taps = taps_per_slot[slot] if slot < len(taps_per_slot) else 1
            state["taps"] = filters[taps]
            state["zi"] = np.zeros(taps - 1)
            state["slot"] = slot
        idx = pos + np.arange(frames)
        outdata[:, 0], state["zi"] = lfilter(state["taps"], 1.0, pattern[idx % slot_samples],
                                             zi=state["zi"])
        if pos + frames <= total:
            recorded[pos:pos + frames] = indata[:, 0]
        if status:
            state["flags"].append(slot)
        while time.perf_counter() - start < load * frames / fs:
            pass
        state["proc_times"].append((slot, (time.perf_counter() - start) * fs / frames))
        state["pos"] = pos + frames

    if fake_delay is None:
        with sd.Stream(samplerate=fs, blocksize=blocksize, channels=1, dtype="float32",
                       latency="low", callback=callback):
            time.sleep(seconds)
    else:
        played = np.zeros(total + fake_delay + blocksize, dtype=np.float32)
        outdata = np.zeros((blocksize, 1), dtype=np.float32)
        noise_rng = np.random.default_rng(0)
        for pos in range(0, total, blocksize):
            indata = played[pos:pos + blocksize, None] + 0.01 * noise_rng.standard_normal((blocksize, 1))
            callback(indata, outdata, blocksize, None, 0)
            played[pos + fake_delay:pos + fake_delay + blocksize] = outdata[:, 0]

    results = []
    proc_times = np.array(state["proc_times"])
    for k in range(len(taps_per_slot)):
        slot_times = proc_times[proc_times[:, 0] == k, 1]
        flags = state["flags"].count(k)
        if (k + 1) * slot_samples > min(state["pos"], total):
            results.append((np.nan, 0.0, flags, np.nan, np.nan))
            continue
        _, delay_ms, ratio = find_delay(
            recorded[k * slot_samples:(k + 1) * slot_samples].astype(np.float64), reference, fs)
        results.append((delay_ms, ratio, flags, slot_times.mean(), slot_times.max()))
    return results


def extra_delays(rows, blocksize):
    """Delay of each valid A chirp minus the median 1-tap delay of its opening,
    as (taps, opening, extra ms)."""
    extras = []
    for opening in sorted({r[3] for r in rows}):
        sel = [r for r in rows if r[0] == "A" and r[1] == blocksize and r[3] == opening and r[8]]
        baseline = [r[5] for r in sel if r[2] == 1]
        if baseline:
            extras += [(r[2], opening, r[5] - np.median(baseline)) for r in sel]
    return extras


os.makedirs("results", exist_ok=True)
rows = []
if mode == "plot":
    with open("results/processing_delay.csv") as f:
        for r in csv.DictReader(f):
            rows.append([r["experiment"], int(r["buffer"]), float(r["condition"]),
                         int(r["opening"]), int(r["chirp"]), float(r["delay_ms"]),
                         float(r["peak_ratio"]), int(r["flags"]), int(r["valid"]),
                         float(r["proc_mean"]), float(r["proc_max"])])
else:
    if mode != "test":
        print_devices()
        print("\nwarm-up: one opening per buffer, not saved")
        for blocksize in BLOCKSIZES:
            run_once(blocksize, [1], 0, SECONDS["B"])
            time.sleep(0.5)
    rng = random.Random(0)
    for opening in range(1, (1 if mode == "test" else rounds) + 1):
        conditions = ([("A", blocksize, 0) for blocksize in BLOCKSIZES]
                      + [("B", blocksize, load) for blocksize in BLOCKSIZES for load in LOADS])
        rng.shuffle(conditions)
        print(f"\n===== round {opening} =====")
        for experiment, blocksize, load in conditions:
            n_chirps = int(SECONDS[experiment] / SLOT) - 2
            if experiment == "A":
                taps_per_slot = TAPS * (n_chirps // len(TAPS))
                rng.shuffle(taps_per_slot)
            else:
                taps_per_slot = [1] * n_chirps
            fake_delay = round(TEST_DELAY[blocksize] / 1000 * fs) if mode == "test" else None
            results = run_once(blocksize, taps_per_slot, load, SECONDS[experiment], fake_delay)
            new_rows = []
            for chirp_index, (taps, (delay_ms, ratio, flags, proc_mean, proc_max)) in enumerate(
                    zip(taps_per_slot, results)):
                new_rows.append([experiment, blocksize, taps if experiment == "A" else load,
                                 opening, chirp_index, round(delay_ms, 4), round(ratio, 1), flags,
                                 int(ratio >= MIN_RATIO), round(proc_mean, 3), round(proc_max, 3)])
            rows += new_rows
            n_valid = sum(r[8] for r in new_rows)
            n_flags = sum(r[7] for r in new_rows)
            if experiment == "A":
                extras = extra_delays(new_rows, blocksize)
                extra_text = "  ".join(
                    f"{taps}: {np.median([extra for chirp_taps, _, extra in extras if chirp_taps == taps]):.2f}"
                    for taps in TAPS[1:] if any(chirp_taps == taps for chirp_taps, _, _ in extras))
                print(f"A  buffer {blocksize:4d}: {n_valid}/{len(new_rows)} chirps, "
                      f"{n_flags} flags, extra ms  {extra_text if extras else 'no 1-tap chirp found'}")
            else:
                good = [r[5] for r in new_rows if r[8]]
                print(f"B  buffer {blocksize:4d}  load {load:.1f}: {n_valid}/{len(new_rows)} chirps, "
                      f"{np.median(good) if good else np.nan:7.2f} ms, {n_flags} flags, processing "
                      f"mean {np.nanmean([r[9] for r in new_rows]):.2f} "
                      f"max {np.nanmax([r[10] for r in new_rows]):.2f}")
            if mode != "test":
                time.sleep(0.5)

    if mode != "test":
        with open("results/processing_delay.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["experiment", "buffer", "condition", "opening", "chirp", "delay_ms",
                        "peak_ratio", "flags", "valid", "proc_mean", "proc_max"])
            w.writerows(rows)

# summary, processing time is a part of the buffer duration
print("\n===== A: FIR length, extra delay = chirp minus the 1-tap median of its opening =====")
print("buffer  taps   predicted ms (samples)   measured ms (samples)   min..max     "
      "spread of repeats   chirps   processing mean / max")
fig, axes = plt.subplot_mosaic([["A", "delay"], ["A", "flags"]], figsize=(12, 6))
marker_style = {256: dict(marker="o", ms=7), 1024: dict(marker="s", ms=11, mfc="none", mew=2)}
for blocksize in BLOCKSIZES:
    extras = extra_delays(rows, blocksize)
    predicted, measured = [], []
    for taps in TAPS:
        extra = np.array([x for chirp_taps, _, x in extras if chirp_taps == taps])
        sel = [r for r in rows if r[0] == "A" and r[1] == blocksize and r[2] == taps]
        if not sel:
            continue
        spread = max((np.ptp([r[5] for r in sel if r[3] == opening and r[8]] or [0])
                      for opening in {r[3] for r in sel}))
        predicted_ms = (taps - 1) / 2 / fs * 1000
        if len(extra):
            predicted.append(predicted_ms)
            measured.append(np.median(extra))
            measured_text = (f"{np.median(extra):7.2f} ({np.median(extra) * fs / 1000:6.0f})"
                             f"   {extra.min():6.2f}..{extra.max():<6.2f}")
        else:
            measured_text = "     no valid chirp              "
        print(f"{blocksize:6d} {taps:5d}   {predicted_ms:7.2f} ({(taps - 1) // 2:4d})          {measured_text}"
              f"   {spread:6.2f} ms        {sum(r[8] for r in sel):2d}/{len(sel):<2d}"
              f"    {np.nanmean([r[9] for r in sel]):.3f} / {np.nanmax([r[10] for r in sel]):.3f}")
    axes["A"].plot(predicted, measured, ls="none", label=f"buffer {blocksize}", **marker_style[blocksize])

print("\n===== B: CPU load =====")
print("buffer  load   median delay ms   min..max of all chirps (ms)   median of each opening (ms)        "
      "flags per opening   chirps   processing mean / max")
jitter = np.random.default_rng(0)
for blocksize, color, offset in zip(BLOCKSIZES, ["C0", "C1"], [-0.03, 0.03]):
    loads, medians, flag_counts = [], [], []
    for load in LOADS:
        sel = [r for r in rows if r[0] == "B" and r[1] == blocksize and r[2] == load]
        if not sel:
            continue
        loads.append(load)
        openings = sorted({r[3] for r in sel})
        good = [r[5] for r in sel if r[8]]
        opening_medians = [np.median([r[5] for r in sel if r[3] == opening and r[8]] or [np.nan])
                           for opening in openings]
        flags = [sum(r[7] for r in sel if r[3] == opening) for opening in openings]
        medians.append(np.median(good) if good else np.nan)
        flag_counts.append(np.mean(flags))
        min_max = f"{min(good):.2f}..{max(good):.2f}" if good else "no valid chirp"
        print(f"{blocksize:6d}  {load:4.1f}   {medians[-1]:8.2f}          {min_max:29s} "
              f"{'  '.join(f'{median:.2f}' for median in opening_medians):34s} "
              f"{' '.join(str(x) for x in flags):18s}  {len(good):2d}/{len(sel):<2d}"
              f"    {np.nanmean([r[9] for r in sel]):.3f} / {np.nanmax([r[10] for r in sel]):.3f}")
        x = load + offset + jitter.uniform(-0.015, 0.015, len(good))
        axes["delay"].plot(x, good, ".", color=color, ms=5, alpha=0.6,
                           label=f"buffer {blocksize}" if len(loads) == 1 else None)
    axes["flags"].plot(loads, flag_counts, lw=2, label=f"buffer {blocksize}", **marker_style[blocksize])

line_end = (TAPS[-1] - 1) / 2 / fs * 1000 * 1.05
axes["A"].plot([0, line_end], [0, line_end], "k--", lw=1, label="measured = predicted")
axes["A"].set_xlabel("Predicted extra delay, (N-1)/2 samples (ms)")
axes["A"].set_ylabel("Measured extra delay (ms)")
axes["A"].set_title("A: FIR filter in the callback")
axes["delay"].set_ylabel("Delay of each chirp (ms)")
axes["delay"].set_title("B: busy-wait in the callback")
axes["flags"].set_ylabel("Flagged callbacks\nper opening")
axes["flags"].set_xlabel("Load (part of the buffer duration)")
axes["flags"].set_ylim(bottom=0)
for name in ["delay", "flags"]:
    axes[name].set_xticks(LOADS)
axes["flags"].set_xlim(axes["delay"].get_xlim())
for ax in axes.values():
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
plt.tight_layout()
if mode != "test":
    plt.savefig("results/processing_delay.png", dpi=150)
    print("\nsaved results/processing_delay.csv and results/processing_delay.png")
    plt.show()
