import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from openwakeword.model import Model
import openwakeword
import ollama

# ---- Config ----
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280          # openwakeword expects 80ms chunks at 16kHz
RECORD_SECONDS = 5         # how long to record after wake word triggers
WAKE_THRESHOLD = 0.5
OLLAMA_MODEL = "llama3.2:3b"
SYSTEM_PROMPT = "You are Jarvis, a helpful voice assistant. Keep responses concise, a sentence or two, since they will be spoken aloud."

# ---- Load models ----
openwakeword.utils.download_models()
print("Loading wake word model...")
wake_model = Model(wakeword_models=["hey_jarvis"])

print("Loading whisper model...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

print("\nReady. Listening for 'Hey Jarvis'...\n")

# ---- State ----
triggered = False

def record_command(duration=RECORD_SECONDS):
    print("Listening for your command...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    segments, _ = whisper_model.transcribe(audio, language="en")
    text = " ".join(seg.text for seg in segments).strip()
    return text

def ask_ollama(prompt):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]

def wake_callback(indata, frames, time, status):
    global triggered
    audio = (indata[:, 0] * 32767).astype(np.int16)
    prediction = wake_model.predict(audio)
    for mdl_name, score in prediction.items():
        if score > WAKE_THRESHOLD:
            triggered = True

# ---- Main loop ----
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                     blocksize=CHUNK_SIZE, callback=wake_callback):
    while True:
        sd.sleep(100)
        if triggered:
            triggered = False
            command_audio = record_command()
            print("Transcribing...")
            text = transcribe(command_audio)
            print(f"You said: {text}")

            if not text.strip():
                print("(nothing heard, going back to listening)\n")
                continue

            print("Thinking...")
            reply = ask_ollama(text)
            print(f"Jarvis: {reply}\n")
            print("Listening for 'Hey Jarvis'...\n")
