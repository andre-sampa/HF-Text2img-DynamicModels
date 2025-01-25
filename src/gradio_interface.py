import gradio as gr


def create_gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            # Prompt input with default value
            prompt = gr.Textbox(
                label="Prompt", 
                value=ImageGenerator.DEFAULT_PROMPT, 
                placeholder="Enter image generation prompt"
            )
            
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