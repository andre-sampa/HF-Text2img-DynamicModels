import random  # Import the random module for generating random seeds
from datetime import datetime
from huggingface_hub import InferenceClient
from IPython.display import display, clear_output
import ipywidgets as widgets
from config.config_colab import api_token
from colab import output

# Step 13: Define the Image Generation Function AA
def generate_image(button, output):
    # Clear the output widget (only clears the image and logs, not the widgets)
    output.clear_output(wait=True)

    # Get the selected model and parameters
    model_id = model_dropdown.value
    prompt = prompt_input.value
    height = height_input.value
    width = width_input.value
    num_inference_steps = num_inference_steps_input.value
    guidance_scale = guidance_scale_input.value

    # Generate a random seed if the checkbox is checked
    if randomise_seed_checkbox.value:
        seed = random.randint(0, 2**32 - 1)  # Generate a random seed
    else:
        seed = seed_input.value  # Use the user-provided seed

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
    with output:
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

            # Save and display the image
            image.save(image_path)
            print(f"Image saved as '{image_path}'")
            print(f"Metadata saved as '{metadata_path}'")
            display(image)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check the model compatibility or try again later.")
