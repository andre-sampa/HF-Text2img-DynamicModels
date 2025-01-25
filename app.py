import os
import json
import random
from datetime import datetime
from huggingface_hub import InferenceClient
from config.config import api_token

# Function to generate an image
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
    if randomise_seed:
        seed = random.randint(0, 2**32 - 1)

    client = InferenceClient(model=model_id, token=api_token)
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = (
        f"{timestamp}_{model_id.split('/')[-1]}_{prompt[:20]}_"
        f"{height}x{width}_steps{num_inference_steps}_scale{guidance_scale}_seed{seed}.png"
    )
    image_path = os.path.join(output_folder, image_name)
    
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

# Entry point for execution if GUI cannot run
def main():
    try:
        prompt = input("Enter a prompt for image generation: ").strip()
        if prompt:
            generate_image(prompt=prompt, randomise_seed=True)
        else:
            print("Please provide a valid prompt.")
    except EOFError:
        print("No input provided. Exiting...")

if __name__ == "__main__":
    try:
        from tkinter import Tk, Button, Label, Entry, StringVar

        root = Tk()
        root.title("Image Generator")

        Label(root, text="Enter prompt:").grid(row=0, column=0, padx=5, pady=5)
        prompt_var = StringVar()
        Entry(root, textvariable=prompt_var, width=40).grid(row=0, column=1, padx=5, pady=5)

        Button(root, text="Generate Image", command=lambda: generate_image(prompt=prompt_var.get(), randomise_seed=True)).grid(row=1, column=0, columnspan=2, pady=10)

        root.mainloop()
    except Exception as e:
        print(f"GUI not available: {e}")
        main()
