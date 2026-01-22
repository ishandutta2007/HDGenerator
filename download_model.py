import urllib.request
import os

url = "https://huggingface.co/svjack/AI-onnx/resolve/main/RealESR_Gx4_fp16.onnx"
output_path = "models/RealESR_Gx4_fp16.onnx"

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print(f"Downloading model from {url}...")

try:
    urllib.request.urlretrieve(url, output_path)
    print(f"Model successfully saved to {output_path}")
except Exception as e:
    print(f"An error occurred: {e}")