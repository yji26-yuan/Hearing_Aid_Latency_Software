"""Device plays a chirp through the speakers, records it with the microphone
   and measures the delay with measure_latency() from step2_wav_delay.py.

Usage:
    python3 step3_real_record.py          (buffer 256)
    python3 step3_real_record.py 512      (buffer 512)
"""
import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import chirp
from step2_wav_delay import measure_latency

blocksize = int(sys.argv[1]) if len(sys.argv) > 1 else 256
os.makedirs("wav", exist_ok=True)

# The stream runs at the output device's sample rate.
# If the input device is set to a different rate, then the system resamples the input.
fs = int(sd.query_devices(kind="output")["default_samplerate"])
print(f"output device: {sd.query_devices(kind='output')['name']}")
print(f"input device:  {sd.query_devices(kind='input')['name']}")
print(f"sample rate: {fs} Hz, buffer: {blocksize} samples")

# test chirp at half scale
duration = 0.5
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
reference = 0.5 * chirp(t, f0=200, f1=8000, t1=duration, method="linear")

# half a second of silence at the end so the delayed chirp fits in the recording
playback = np.concatenate([reference, np.zeros(int(fs * 0.5))])

recorded = sd.playrec(playback, samplerate=fs, channels=1,
                      blocksize=blocksize, latency="low")
sd.wait()
if sd.get_status():
    print(f"warning: the audio glitched ({sd.get_status()}), the delay may be off")

sf.write("wav/reference_real.wav", reference, fs)
sf.write("wav/recorded_real.wav", recorded, fs)
print("peak level of the recording:", float(np.max(np.abs(recorded))))

measure_latency("wav/reference_real.wav", "wav/recorded_real.wav",
                save_as=f"results/fig3_single_measurement_{blocksize}.png")