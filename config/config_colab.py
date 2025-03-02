from google.colab import userdata

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
print("=================================")