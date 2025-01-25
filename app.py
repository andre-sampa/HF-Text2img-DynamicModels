import os
import sys

# Set PYTHONPATH to the project root 
# Solves all problems w subfolders - ONLY WAY FOR IPYNB FILES
os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(".."))

import json
import random
from datetime import datetime
from huggingface_hub import InferenceClient

# Import API token from config
from config.config import api_token

# Step 4: Define Model Options
MODELS = [
    ("stable-diffusion-3.5-large", "stabilityai/stable-diffusion-3.5-large"),
    ("Stable Diffusion 3.5-large-turbo", "stabilityai/stable-diffusion-3.5-large-turbo"),
    ("Stable Diffusion 2.1", "stabilityai/stable-diffusion-2-1"),
    ("Stable Diffusion 1.5", "runwayml/stable-diffusion-v1-5"),
    ("FLUX1.0", "black-forest-labs/FLUX.1-dev"),
    ("Flux-Midjourney-Mix2-LoRA", "strangerzonehf/Flux-Midjourney-Mix2-LoRA"),
    ("Custom Model", "your-custom-model-id"),  # Add your own model here
]

# Step 5: Define Default Parameters
DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1"
DEFAULT_PROMPT = "Astronaut riding a horse"
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 42
DEFAULT_RANDOMISE_SEED = False

# Step 6: Define the Image Generation Function
def generate_image(model_id, prompt, height, width, num_inference_steps, guidance_scale, seed, randomise_seed):
    # Generate a random seed if the checkbox is checked
    if randomise_seed:
        seed = random.randint(0, 2**32 - 1)  # Generate a random seed

    # Initialize the InferenceClient
    client = InferenceClient(model=model_id, token=api_token)

    # Create the output folder if it doesn't exist
    output_folder = "generated_images"
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique image name with timestamp at the start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{timestamp}_{model_id.split('/')[-1]}_{prompt[:20]}_{height}x{width}_steps{num_inference_steps}_scale{guidance_scale}_seed{seed}.png"
    image_path = os.path.join(output_folder, image_name)

    # Save metadata as a JSON file with the same name as the image
    metadata_name = image_name.replace(".png", ".json")
    metadata_path = os.path.join(output_folder, metadata_name)

    metadata = {
        "model": model_id,
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "image_path": image_path,
        "timestamp": timestamp
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Generate the image with additional parameters
    print(f"Generating image for prompt: '{prompt}' using model: {model_id}...")
    try:
        image = client.text_to_image(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        image.save(image_path)
        print(f"Image saved as '{image_path}'")
        print(f"Metadata saved as '{metadata_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the model compatibility or try again later.")

# Step 7: Main Execution Block
if __name__ == "__main__":
    # Get user input (replace with command-line arguments or GUI inputs)
    model_id = DEFAULT_MODEL
    prompt = DEFAULT_PROMPT
    height = DEFAULT_HEIGHT
    width = DEFAULT_WIDTH
    num_inference_steps = DEFAULT_STEPS
    guidance_scale = DEFAULT_GUIDANCE_SCALE
    seed = DEFAULT_SEED
    randomise_seed = DEFAULT_RANDOMISE_SEED

    # Generate the image
    generate_image(model_id, prompt, height, width, num_inference_steps, guidance_scale, seed, randomise_seed)