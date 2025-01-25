import os
import json
import random
from datetime import datetime
import PySimpleGUIWeb as sg
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

class ImageGeneratorConfig:
    MODELS = [
        ("Stable Diffusion 3.5-large", "stabilityai/stable-diffusion-3.5-large"),
        ("Stable Diffusion 3.5-large-turbo", "stabilityai/stable-diffusion-3.5-large-turbo"),
        ("Stable Diffusion 2.1", "stabilityai/stable-diffusion-2-1"),
        ("Stable Diffusion 1.5", "runwayml/stable-diffusion-v1-5"),
        ("FLUX1.0", "black-forest-labs/FLUX.1-dev"),
        ("Flux-Midjourney-Mix2-LoRA", "strangerzonehf/Flux-Midjourney-Mix2-LoRA"),
        ("Custom Model", "your-custom-model-id")
    ]
    DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1"
    DEFAULT_PROMPT = "Astronaut riding a horse"
    DEFAULT_HEIGHT = 1024
    DEFAULT_WIDTH = 1024
    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE = 7.5
    DEFAULT_SEED = 42

def create_gui():
    sg.theme('DarkBlue')

    model_options = [model[0] for model in ImageGeneratorConfig.MODELS]
    model_values = [model[1] for model in ImageGeneratorConfig.MODELS]

    layout = [
        [sg.Text("Select Model:"), 
         sg.Combo(model_options, default_value=ImageGeneratorConfig.DEFAULT_MODEL, key="MODEL", size=(40, 1))],
        [sg.Text("Prompt:"), 
         sg.Multiline(default_text=ImageGeneratorConfig.DEFAULT_PROMPT, key="PROMPT", size=(50, 3))],
        [sg.Text("Height:"), 
         sg.Slider(range=(64, 1024), default_value=ImageGeneratorConfig.DEFAULT_HEIGHT, orientation='h', key="HEIGHT", size=(40, 15))],
        [sg.Text("Width:"), 
         sg.Slider(range=(64, 1024), default_value=ImageGeneratorConfig.DEFAULT_WIDTH, orientation='h', key="WIDTH", size=(40, 15))],
        [sg.Text("Inference Steps:"), 
         sg.Slider(range=(1, 100), default_value=ImageGeneratorConfig.DEFAULT_STEPS, orientation='h', key="STEPS", size=(40, 15))],
        [sg.Text("Guidance Scale:"), 
         sg.Slider(range=(0.0, 20.0), default_value=ImageGeneratorConfig.DEFAULT_GUIDANCE, orientation='h', key="SCALE", size=(40, 15))],
        [sg.Text("Seed:"), 
         sg.Input(default_text=str(ImageGeneratorConfig.DEFAULT_SEED), key="SEED", size=(20, 1)),
         sg.Checkbox("Randomize Seed", key="RANDOMIZE")],
        [sg.Button("Generate Image"), sg.Button("Exit")]
    ]

    return sg.Window("AI Image Generator", layout, finalize=True)

def generate_image(model_id, prompt, height, width, steps, scale, seed, randomize_seed):
    load_dotenv()
    api_token = os.getenv('HF_TOKEN')
    
    if not api_token:
        raise ValueError("Hugging Face API token not found.")

    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)

    output_folder = "generated_images"
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{timestamp}_{model_id.split('/')[-1]}_{prompt[:20]}_{height}x{width}_steps{steps}_scale{scale}_seed{seed}.png"
    image_path = os.path.join(output_folder, image_name)

    client = InferenceClient(model=model_id, token=api_token)

    metadata = {
        "model": model_id,
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": scale,
        "seed": seed,
        "image_path": image_path,
        "timestamp": timestamp
    }

    try:
        image = client.text_to_image(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=scale,
            seed=seed
        )
        
        image.save(image_path)
        with open(image_path.replace('.png', '.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        return image_path
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")

def main():
    window = create_gui()

    while True:
        event, values = window.read()

        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Generate Image":
            try:
                selected_model_index = window["MODEL"].get_index()
                model_id = ImageGeneratorConfig.MODELS[selected_model_index][1]

                image_path = generate_image(
                    model_id=model_id,
                    prompt=values["PROMPT"],
                    height=int(values["HEIGHT"]),
                    width=int(values["WIDTH"]),
                    steps=int(values["STEPS"]),
                    scale=float(values["SCALE"]),
                    seed=int(values["SEED"]) if values["SEED"] else ImageGeneratorConfig.DEFAULT_SEED,
                    randomize_seed=values["RANDOMIZE"]
                )
                sg.popup_ok(f"Image generated: {image_path}")
            except Exception as e:
                sg.popup_error(f"Generation failed: {e}")

    window.close()

if __name__ == "__main__":
    main()