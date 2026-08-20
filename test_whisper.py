import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 5  # seconds

print("Loading whisper model (small, CPU)...")
model = WhisperModel("small", device="cpu", compute_type="int8")

print(f"\nRecording for {DURATION} seconds... speak now!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
print("Recording done. Transcribing...")

audio = audio.flatten()
segments, info = model.transcribe(audio, language="en")

print(f"\nDetected language: {info.language} (confidence: {info.language_probability:.2f})")
print("\nTranscript:")
for segment in segments:
    print(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
