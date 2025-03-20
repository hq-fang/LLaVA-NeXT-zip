from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image, ImageDraw
import requests
import copy
import torch
import sys
import warnings
import logging
from threading import Thread
from transformers import TextIteratorStreamer
import json
import random


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
pretrained = "/net/nfs/prior/jiafei/unified_VLM/LLaVA-NeXT/checkpoints/onevision/RoboPoint_Linedraw_05b_3/checkpoint-60000"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

# Load the image (we'll use the same image for each prompt)
local_image_path = "/net/nfs/prior/jiafei/unified_VLM/LLaVA-NeXT/checkpoints/test_image/light.png"

# Open the image from the local file path
image = Image.open(local_image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
image_sizes = [image.size]

# Function to process the prompt and generate output
def process_prompt(user_prompt):
    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN + f"\n{user_prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    max_context_length = getattr(model.config, "max_position_embeddings", 2048)
    num_image_tokens = question.count(DEFAULT_IMAGE_TOKEN) * model.get_vision_tower().num_patches

    # Streamer setup
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
    max_new_tokens = min(4096, max_context_length - input_ids.shape[-1] - num_image_tokens)

    # Handle cases when token limit is exceeded
    if max_new_tokens < 1:
        print(
            json.dumps(
                {
                    "text": question + "Exceeds max token length. Please start a new conversation, thanks.",
                    "error_code": 0,
                }
            )
        )
        return None
    else:
        gen_kwargs = {
            "do_sample": False,
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "images": image_tensor,
            "image_sizes": image_sizes,
        }

        thread = Thread(
            target=model.generate,
            kwargs=dict(
                inputs=input_ids,
                streamer=streamer,
                **gen_kwargs,
            ),
        )
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            print(generated_text, flush=True)

        return generated_text

# Function to visualize coordinates on the image
def visualize_coordinates_from_output(generated_text, image):
    coordinates = []
    image_width, image_height = image.size

    if "[" in generated_text and "]" in generated_text:  # Check if there is a list of coordinates
        generated_text = generated_text.strip("[]")  # Remove brackets
        coord_pairs = generated_text.split("), (")  # Split coordinate pairs
        for pair in coord_pairs:
            x_str, y_str = pair.strip("()").split(", ")
            x, y = float(x_str), float(y_str)
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(x * image_width)
            pixel_y = int(y * image_height)
            coordinates.append((pixel_x, pixel_y))

    if coordinates:
        def visualize_2d(img, points, cross_size=9, cross_width=4):
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            draw = ImageDraw.Draw(img)
            size = cross_size
            width = cross_width

            # Choose a random color once per prompt
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            for x, y in points:
                draw.line((
                    x - size, y - size, x + size, y + size
                ), fill=random_color, width=width)
                draw.line((
                    x - size, y + size, x + size, y - size
                ), fill=random_color, width=width)

            img = img.convert('RGB')
            return img

        # Draw the coordinates on the image
        return visualize_2d(image, coordinates)
    else:
        print("No coordinates found in model output.")
        return image

# Main loop to process user prompts and re-display the same image
while True:
    # Get user prompt from the terminal
    user_prompt = input("Please enter the prompt (or type 'exit' to quit): ")
    if user_prompt.lower() == 'exit':
        break

    # Process the prompt and generate output
    generated_text = process_prompt(user_prompt)

    # Visualize the coordinates (if any) and display the image
    if generated_text:
        processed_image = visualize_coordinates_from_output(generated_text, image)
        processed_image.show()  # Show the image with or without drawn points
        processed_image.save('/net/nfs/prior/jiafei/unified_VLM/LLaVA-NeXT/checkpoints/test_image/output_image.png')  # Save the image with points drawn