import os
import json
import random
from datetime import datetime
from huggingface_hub import InferenceClient
from config.config import api_token
import PySimpleGUI as sg

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

    sg.popup(f"Generating image for prompt: '{prompt}' using model: {model_id}...")
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
        sg.popup(f"Image saved as '{image_path}'", f"Metadata saved as '{metadata_path}'")
    except Exception as e:
        sg.popup_error(f"An error occurred: {e}", "Please check the model compatibility or try again later.")

# GUI interface
def main():
    sg.theme("DarkBlue")

    layout = [
        [sg.Text("Enter Prompt:"), sg.InputText(key="PROMPT")],
        [sg.Text("Model ID:"), sg.InputText("stabilityai/stable-diffusion-2-1", key="MODEL_ID")],
        [sg.Text("Image Height:"), sg.InputText("1024", key="HEIGHT")],
        [sg.Text("Image Width:"), sg.InputText("1024", key="WIDTH")],
        [sg.Text("Inference Steps:"), sg.InputText("50", key="STEPS")],
        [sg.Text("Guidance Scale:"), sg.InputText("7.5", key="SCALE")],
        [sg.Text("Seed (leave empty for random):"), sg.InputText("", key="SEED")],
        [sg.Text("Output Folder:"), sg.InputText("generated_images", key="OUTPUT")],
        [sg.Checkbox("Randomize Seed", default=False, key="RANDOMIZE")],
        [sg.Button("Generate"), sg.Button("Exit")]
    ]

    window = sg.Window("Image Generator", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        if event == "Generate":
            prompt = values["PROMPT"]
            if not prompt:
                sg.popup_error("Prompt cannot be empty!")
                continue

            try:
                seed = int(values["SEED"]) if values["SEED"] else 42
                generate_image(
                    model_id=values["MODEL_ID"],
                    prompt=prompt,
                    height=int(values["HEIGHT"]),
                    width=int(values["WIDTH"]),
                    num_inference_steps=int(values["STEPS"]),
                    guidance_scale=float(values["SCALE"]),
                    seed=seed,
                    randomise_seed=values["RANDOMIZE"],
                    output_folder=values["OUTPUT"]
                )
            except ValueError as ve:
                sg.popup_error(f"Invalid input: {ve}")

    window.close()

if __name__ == "__main__":
    main()
