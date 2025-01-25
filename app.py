import os
import json
import random
from datetime import datetime
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

class ImageGenerator:
    MODELS = [
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-large-turbo", 
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
        "black-forest-labs/FLUX.1-dev"
    ]

    @staticmethod
    def generate_image(
        model_id, 
        prompt, 
        height=1024, 
        width=1024, 
        steps=50, 
        scale=7.5, 
        seed=None,
        randomize_seed=False
    ):
        # Load API token
        load_dotenv()
        api_token = os.getenv('HF_TOKEN')
        
        if not api_token:
            raise ValueError("Hugging Face API token not found")

        # Randomize seed if requested
        if randomize_seed or seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Create output directory
        output_folder = "generated_images"
        os.makedirs(output_folder, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = (
            f"{timestamp}_{model_id.split('/')[-1]}_{prompt[:20]}_"
            f"{height}x{width}_steps{steps}_scale{scale}_seed{seed}.png"
        )
        image_path = os.path.join(output_folder, image_name)

        # Initialize client
        client = InferenceClient(model=model_id, token=api_token)

        # Prepare metadata
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

        # Generate and save image
        image = client.text_to_image(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=scale,
            seed=seed
        )
        
        image.save(image_path)
        
        # Save metadata
        with open(image_path.replace('.png', '.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        return image_path

def create_gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            # Prompt input
            prompt = gr.Textbox(label="Prompt", placeholder="Enter image generation prompt")
            
            # Model selection
            model = gr.Dropdown(
                choices=ImageGenerator.MODELS, 
                value="stabilityai/stable-diffusion-2-1", 
                label="Select Model"
            )
        
        with gr.Row():
            # Numeric inputs
            height = gr.Slider(minimum=64, maximum=1024, value=1024, label="Height")
            width = gr.Slider(minimum=64, maximum=1024, value=1024, label="Width")
        
        with gr.Row():
            steps = gr.Slider(minimum=1, maximum=100, value=50, label="Inference Steps")
            scale = gr.Slider(minimum=0.0, maximum=20.0, value=7.5, label="Guidance Scale")
        
        with gr.Row():
            seed = gr.Number(value=42, label="Seed")
            randomize_seed = gr.Checkbox(label="Randomize Seed")
        
        # Generate button
        generate_btn = gr.Button("Generate Image")
        
        # Output image
        output_image = gr.Image(label="Generated Image")
        
        # Generation logic
        generate_btn.click(
            fn=ImageGenerator.generate_image, 
            inputs=[model, prompt, height, width, steps, scale, seed, randomize_seed],
            outputs=output_image
        )

    return demo

def main():
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()