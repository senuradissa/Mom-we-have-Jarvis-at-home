import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from openwakeword.model import Model
import openwakeword
import ollama
from piper import PiperVoice

#This section will sort out the models. There will be a significant delay on the first time of opening the program (About 30s)
# ---- Config ----
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280          # openwakeword expects 80ms chunks at 16kHz
RECORD_SECONDS = 5         # how long to record after wake word triggers
WAKE_THRESHOLD = 0.5
OLLAMA_MODEL = "llama3.2:3b"
VOICE_PATH = "voices/en_US-ryan-medium.onnx"
SYSTEM_PROMPT = "You are Jarvis, a helpful voice assistant. Keep responses concise, a sentence or two, since they will be spoken aloud."

# ---- Load models ----
openwakeword.utils.download_models()
print("Loading wake word model...")
wake_model = Model(wakeword_models=["hey_jarvis"])

print("Loading whisper model...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

print("Loading voice model...")
tts_voice = PiperVoice.load(VOICE_PATH)

print("\nReady. Listening for 'Hey Jarvis'...\n")

#These functions will deal with sorting out the recording, transcribing and then speaking of the ai models response to the user's query.
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

def speak(text):
    chunks = []
    sample_rate = None
    for chunk in tts_voice.synthesize(text):
        sample_rate = chunk.sample_rate
        chunks.append(chunk.audio_float_array)
    audio = np.concatenate(chunks)

    #Adding some padding since it likes to cut off the last word or so. About 300ms should do for now
    pad_sample = int(0.3*sample_rate)
    silence = np.zeros(pad_sample, dtype=audio.dtype)
    audio = np.concatenate([silence, audio, silence])

    sd.play(audio, samplerate=sample_rate)
    sd.wait()

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
            print(f"Jarvis: {reply}")
            speak(reply)
            print("\nListening for 'Hey Jarvis'...\n")
