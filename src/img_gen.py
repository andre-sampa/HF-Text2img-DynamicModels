from IPython.display import display, clear_output
import random  # Import the random module for generating random seeds
from datetime import datetime
from IPython.display import display, clear_output
import ipywidgets as widgets

# Step 4: Create a Dropdown for Model Selection (with wider layout)
model_dropdown = widgets.Dropdown(
    options=[
        ("stable-diffusion-3.5-large", "stabilityai/stable-diffusion-3.5-large"),
        ("Stable Diffusion 3.5-large-turbo", "stabilityai/stable-diffusion-3.5-large-turbo"),
        ("Stable Diffusion 2.1", "stabilityai/stable-diffusion-2-1"),
        ("Stable Diffusion 1.5", "runwayml/stable-diffusion-v1-5"),
        ("FLUX1.0", "black-forest-labs/FLUX.1-dev"),
        ("Flux-Midjourney-Mix2-LoRA", "strangerzonehf/Flux-Midjourney-Mix2-LoRA"),
        ("Custom Model", "your-custom-model-id"),  # Add your own model here
    ],
    value="stabilityai/stable-diffusion-2-1",  # Default model
    description="Select Model:",
    disabled=False,
    layout=widgets.Layout(width="500px")  # Set the width of the dropdown
)

# Step 5: Create a Text Input for the Prompt (with wider layout)
prompt_input = widgets.Textarea(
    value="Astronaut riding a horse",  # Default prompt
    placeholder="Enter your prompt here...",
    description="Prompt:",
    disabled=False,
    layout=widgets.Layout(width="500px", height="100px")  # Set the width and height of the text input
)

# Step 6: Create Number Inputs for Height and Width
height_input = widgets.IntSlider(
    value=1024,  # Default height
    min=64,  # Minimum height
    max=1024,  # Maximum height
    step=64,  # Step size
    description="Height:",
    disabled=False,
    layout=widgets.Layout(width="500px")
)

width_input = widgets.IntSlider(
    value=1024,  # Default width
    min=64,  # Minimum width
    max=1024,  # Maximum width
    step=64,  # Step size
    description="Width:",
    disabled=False,
    layout=widgets.Layout(width="500px")
)

# Step 7: Create Number Inputs for Inference Steps and Guidance Scale
num_inference_steps_input = widgets.IntSlider(
    value=50,  # Default number of inference steps
    min=1,  # Minimum steps
    max=100,  # Maximum steps
    step=1,  # Step size
    description="Inference Steps:",
    disabled=False,
    layout=widgets.Layout(width="500px")
)

guidance_scale_input = widgets.FloatSlider(
    value=7.5,  # Default guidance scale
    min=0.0,  # Minimum guidance scale
    max=20.0,  # Maximum guidance scale
    step=0.5,  # Step size
    description="Guidance Scale:",
    disabled=False,
    layout=widgets.Layout(width="500px")
)

# Step 8: Create a Number Input for Seed
seed_input = widgets.IntText(
    value=42,  # Default seed
    description="Seed:",
    disabled=False,
    layout=widgets.Layout(width="500px")
)

# Step 9: Add a Checkbox for Randomising the Seed
randomise_seed_checkbox = widgets.Checkbox(
    value=False,  # Default value (unchecked)
    description="Randomise Seed",
    disabled=False,
    layout=widgets.Layout(width="500px")
)

# Step 10: Create a Button to Generate the Image
generate_button = widgets.Button(description="Generate Image")

# Step 11: Create an Output Widget to Display Results
output = widgets.Output()

# Step 12: Define a Function to Handle Checkbox Changes
def handle_randomise_seed(change):
    """
    Enable or disable the seed input based on the checkbox state.
    """
    if change["new"]:  # If checkbox is checked
        seed_input.disabled = True  # Disable the seed input
    else:  # If checkbox is unchecked
        seed_input.disabled = False  # Enable the seed input

# Link the checkbox to the handler function
randomise_seed_checkbox.observe(handle_randomise_seed, names="value")

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

# Step 14: Link the Button to the Function
generate_button.on_click(generate_image)

# Step 15: Display the Widgets
display(model_dropdown)
display(prompt_input)
display(height_input)
display(width_input)
display(num_inference_steps_input)
display(guidance_scale_input)
display(seed_input)
display(randomise_seed_checkbox)  # Display the randomise seed checkbox
display(generate_button)
display(output)  # Display the output widget