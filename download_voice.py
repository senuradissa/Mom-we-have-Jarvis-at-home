import urllib.request
import os

#Medium Quality, just a test voice for the time being.
VOICE_NAME = "en_US-ryan-medium"
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium"

FILES = [
    f"{VOICE_NAME}.onnx",
    f"{VOICE_NAME}.onnx.json",
]

os.makedirs("voices", exist_ok=True)

for fname in FILES:
    dest = os.path.join("voices", fname)
    if os.path.exists(dest):
        print(f"{fname} already exists, skipping.")
        continue
    url = f"{BASE_URL}/{fname}"
    print(f"Downloading {fname}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")

print("\nDone. Voice files are in the 'voices' folder.")
