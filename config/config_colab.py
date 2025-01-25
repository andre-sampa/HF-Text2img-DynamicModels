# config_colab.py
import sys
import os

# Add the src folder to the Python path
# Solves all problems w subfolders - option2
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)
    
from google.colab import userdata
from config.prompts import prompts  # Import prompts from prompts.py

# Retrieve the Hugging Face token from Colab secrets
api_token = userdata.get("HF_TOKEN")

# Debugging: Check if the Hugging Face token is available
if not api_token:
    print("=== Debug: Error ===")
    print("ERROR: Hugging Face token (HF_CTB_TOKEN) is missing. Please set it in Colab secrets.")
else:
    print("=== Debug: Success ===")
    print("Hugging Face token loaded successfully.")

# Debugging: Print prompt and model options
print("=== Debug: Available Options ===")
print("Prompt Options:", [p["alias"] for p in prompts])
print("Model Options:", [m["alias"] for m in models])
print("=================================")