import os
import json
import random
from datetime import datetime
from huggingface_hub import InferenceClient
from config.config import api_token
from tkinter import Tk, Button, Label, Entry, StringVar

def generate_image(
    model_id="stabilityai/stable-diffusion-2-1",
    prompt="Astronaut riding a horse",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    randomise_seed=False,
    output_folder="generated_images"
):
    """
    Generate an image using a selected model and parameters.

    Args:
        model_id (str): The model ID to use for image generation.
        prompt (str): The text prompt for the image.
        height (int): Height of the generated image.
        width (int): Width of the generated image.
        num_inference_steps (int): Number of inference steps.
        guidance_scale (float): Guidance scale value.
        seed (int): Seed for reproducibility.
        randomise_seed (bool): Whether to randomise the seed.
        output_folder (str): Folder to save the generated image and metadata.
    """
    # Generate a random seed if required
    if randomise_seed:
        seed = random.randint(0, 2**32 - 1)

    # Initialize the InferenceClient
    client = InferenceClient(model=model_id, token=api_token)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique image name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = (
        f"{timestamp}_{model_id.split('/')[-1]}_{prompt[:20]}_"
        f"{height}x{width}_steps{num_inference_steps}_scale{guidance_scale}_seed{seed}.png"
    )
    image_path = os.path.join(output_folder, image_name)

    # Save metadata as a JSON file
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
        "timestamp": timestamp,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Generate the image
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

        # Save the image
        image.save(image_path)
        print(f"Image saved as '{image_path}'")
        print(f"Metadata saved as '{metadata_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the model compatibility or try again later.")

def on_generate_button_click():
    prompt = prompt_var.get()
    if prompt:
        generate_image(prompt=prompt, randomise_seed=True)
    else:
        print("Please enter a prompt.")

# Create a simple GUI
root = Tk()
root.title("Image Generator")

Label(root, text="Enter prompt:").grid(row=0, column=0, padx=5, pady=5)
prompt_var = StringVar()
Entry(root, textvariable=prompt_var, width=40).grid(row=0, column=1, padx=5, pady=5)

Button(root, text="Generate Image", command=on_generate_button_click).grid(row=1, column=0, columnspan=2, pady=10)

root.mainloop()
