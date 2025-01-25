


class ImageGenerator:
    DEFAULT_PROMPT = "Astronaut riding a horse"
    MODELS = [
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-large-turbo", 
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
        "black-forest-labs/FLUX.1-dev",
        "strangerzonehf/Flux-Midjourney-Mix2-LoRA",
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