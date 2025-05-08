import requests, zipfile, io, os

url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
output_dir = 'voice_model'

print("Downloading Vosk model...")
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(".")

# Rename folder
extracted_folder = "vosk-model-small-en-us-0.15"
if os.path.exists(extracted_folder):
    os.rename(extracted_folder, output_dir)

print("Model downloaded and ready in './model'")
