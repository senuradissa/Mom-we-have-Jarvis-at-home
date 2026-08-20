from piper import PiperVoice
import sounddevice as sd
import numpy as np

VOICE_PATH = "voices/en_US-ryan-medium.onnx"

print("Loading voice model...")
voice = PiperVoice.load(VOICE_PATH)

text = "Hey, this is Jarvis. Text to speech is working."
print(f"Speaking: {text}")

# synthesize() yields one AudioChunk per sentence
chunks = []
sample_rate = None
for chunk in voice.synthesize(text):
    sample_rate = chunk.sample_rate
    chunks.append(chunk.audio_float_array)

audio = np.concatenate(chunks)
sd.play(audio, samplerate=sample_rate)
sd.wait()

print("Done.")
