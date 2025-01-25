import os
import json
import random
from datetime import datetime
import gradio as gr
from huggingface_hub import InferenceClient
from config.config import api_token
from src.gradio_interface import create_gradio_interface



def main():
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()