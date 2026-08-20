import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from openwakeword.model import Model
import openwakeword
import ollama
from piper import PiperVoice
import datetime
import requests
import os
import json
import subprocess

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

# ---- Tool functions (the actual Python code that DOES things) ----

def get_current_time():
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y, %I:%M %p")

def get_weather(city: str):
    """Fetches current weather for a given city using the free Open-Meteo API (no key needed)."""
    # Step 1: turn city name into lat/lon coordinates
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10)
    geo_data = geo_resp.json()

    if "results" not in geo_data or not geo_data["results"]:
        return f"Couldn't find a location called {city}."

    location = geo_data["results"][0]
    lat, lon = location["latitude"], location["longitude"]

    # Step 2: fetch current weather for those coordinates
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = requests.get(weather_url, params={
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius",
    }, timeout=10)
    weather_data = weather_resp.json()

    temp = weather_data["current"]["temperature_2m"]
    return f"It's currently {temp}°C in {city}."

# Map of nicknames -> actual paths/commands to open.
# Add your own programs/files here as needed.
OPENABLE_APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "vs code": r"C:\Users\Senura Dissanayake\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd",
    "vivado": r"D:\Vivado_Vitis\Vivado\2023.1\bin\vivado.bat",
}

def open_app(app_name: str):
    """Opens a known program by name."""
    key = app_name.lower().strip()
    if key not in OPENABLE_APPS:
        return f"I don't have a saved path for '{app_name}' yet. Known apps: {', '.join(OPENABLE_APPS)}."
    path = OPENABLE_APPS[key]
    try:
        subprocess.Popen(path, shell=True)
        return f"Opening {app_name}."
    except FileNotFoundError:
        return f"The saved path for {app_name} doesn't exist: {path}"
    except Exception as e:
        return f"Failed to open {app_name}. Error: {e}"

# ---- Tool schemas (what we tell the LLM is available) ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name, e.g. 'Ottawa'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": f"Open a program on the PC. Known apps: {', '.join(OPENABLE_APPS)}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the app to open"}
                },
                "required": ["app_name"],
            },
        },
    },
]

# Maps tool name (string) -> actual Python function to call
AVAILABLE_FUNCTIONS = {
    "get_current_time": lambda: get_current_time(),
    "get_weather": lambda city: get_weather(city),
    "open_app": lambda app_name: open_app(app_name),
}

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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # First call: give the model the prompt + list of available tools.
    # The model will either reply normally, OR ask to call one/more tools.
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages, tools=TOOLS)
    messages.append(response["message"])

    tool_calls = response["message"].get("tool_calls")

    if tool_calls:
        # The model wants to call one or more tools. We actually run them here.
        for call in tool_calls:
            func_name = call["function"]["name"]
            func_args = call["function"]["arguments"]
            print(f"  -> Calling tool: {func_name}({func_args})")

            if func_name in AVAILABLE_FUNCTIONS:
                result = AVAILABLE_FUNCTIONS[func_name](**func_args)
            else:
                result = f"Unknown tool: {func_name}"

            # Feed the tool's real result back to the model as a new message
            messages.append({"role": "tool", "content": str(result)})

        # Second call: now the model has the real tool result and can
        # form its final natural-language answer.
        final_response = ollama.chat(model=OLLAMA_MODEL, messages=messages, tools=TOOLS)
        return final_response["message"]["content"]

    # No tool needed, model answered directly
    return response["message"]["content"]

def speak(text):
    chunks = []
    sample_rate = None
    for chunk in tts_voice.synthesize(text):
        sample_rate = chunk.sample_rate
        chunks.append(chunk.audio_float_array)
    audio = np.concatenate(chunks)

    # Pad a bit of trailing (and leading) silence so playback doesn't
    # get cut off right at the end of the last word.
    pad_samples = int(0.3 * sample_rate)  # 300ms
    silence = np.zeros(pad_samples, dtype=audio.dtype)
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
