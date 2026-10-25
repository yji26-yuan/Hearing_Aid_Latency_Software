"""Clap-response test: the computer listens and answers a clap with a short
chirp from the speakers.

Record this with a phone placed 20 cm next to the laptop. The phone recording then
contains both the clap and the answer, so the time between them is the
microphone to speaker response time, measured by a second device.

Usage:
    python3 step6_clap_response.py [buffer]   
"""
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.signal import chirp

BLOCKSIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 256
THRESHOLD = 0.1     # The largest sample in one buffer that counts as a clap
COOLDOWN = 1.5      # 1.5s cooldown after answering

fs = int(sd.query_devices(kind="output")["default_samplerate"])

# The "beep": a 30 ms chirp from 2 to 6 kHz
# rebuilds the same signal, keep the two in sync.
n_beep = int(fs * 0.03)
t_beep = np.linspace(0, 0.03, n_beep, endpoint=False)
beep = (0.8 * chirp(t_beep, f0=2000, f1=6000, t1=0.03, method="linear")
        * np.hanning(n_beep)).astype(np.float32)

# pos:   next sample of the beep to play
# last:  time.monotonic() of the last answer
# count: claps answered
state = {"pos": None, "last": -10.0, "count": 0}


def callback(indata, outdata, frames, time_info, status):
    outdata[:] = 0
    if state["pos"] is None:
        now = time.monotonic()
        if np.max(np.abs(indata)) > THRESHOLD and now - state["last"] > COOLDOWN:
            state["pos"] = 0
            state["last"] = now
            state["count"] += 1
    if state["pos"] is not None:
        start = state["pos"]
        chunk = beep[start:start + frames]
        outdata[:len(chunk), 0] = chunk
        state["pos"] = start + frames
        if state["pos"] >= len(beep):
            state["pos"] = None


print(f"sample rate {fs} Hz, buffer {BLOCKSIZE}. Clap and the computer answers. Ctrl+C to stop.")
with sd.Stream(samplerate=fs, blocksize=BLOCKSIZE, channels=1,
               dtype="float32", latency="low", callback=callback):
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print(f"\nanswered {state['count']} claps")