from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import numpy as np
import cv2
import copy
import torch
import warnings
import re

warnings.filterwarnings("ignore")

# Load the pretrained model
pretrained = "/data/input/jiafei/LLaVA-NeXT/checkpoints/onevision/new_point_7b/checkpoint-12000"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
            "multimodal": True,
                "attn_implementation": "sdpa",
                }
overwrite_config = {}
overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
llava_model_args["overwrite_config"] = overwrite_config
#tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
#tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,ignore_mismatched_sizes=True)
model.eval()

# Load and preprocess the image
local_image_path = "/data/input/jiafei/LLaVA-NeXT/light.png"
image = Image.open(local_image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# Define the prompt
conv_template = "qwen_1_5"
question = (
        DEFAULT_IMAGE_TOKEN + rties of objects to actionable outcomes, such as grasping or placing, and is often represented using keypoints, part segmentation, or dense feature descriptors.\n\n# Point-Based Action Space\n\nThe action space in this task is defined by **2D coaoe, there is an item encased within a red rectangle. Pinpoint several points within the vacant space situated to the left of the object that is highlighted. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the:q:
        :conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel location")
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

# Tokenize and generate predictions
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print("Generated Trajectory:", text_outputs)


# Function to parse the trajectory
def parse_trajectory(text_output):
    if not text_output:
        return []
    raw_string = text_output[0].strip("[]")
    pattern = r"\((\d*\.\d+), (\d*\.\d+)\)"
    coordinates = [(float(x), float(y)) for x, y in re.findall(pattern, raw_string)]
    return coordinates

# Function to extract and denormalize coordinates
def extract_and_denormalize_coordinates(coordinates, image_width, image_height):
    pixel_coordinates = [
        (int(x * image_width), int((1 - y) * image_height)) for x, y in coordinates
    ]
    return pixel_coordinates

# Draw the trajectory on the image
def draw_trajectory_with_color_transition(image, coordinates):
    start_color = np.array([255, 255, 255])  # White
    end_color = np.array([0, 255, 0])  # Green
    trajectory_len = len(coordinates)
    annotated_image = np.array(image)
    for i in range(1, trajectory_len):
        start_point = coordinates[i - 1]
        end_point = coordinates[i]
        color_ratio = i / (trajectory_len - 1)
        line_color = (1 - color_ratio) * start_color + color_ratio * end_color
        line_color = tuple(map(int, line_color))
        cv2.line(annotated_image, start_point, end_point, line_color, 5)
    cv2.circle(annotated_image, coordinates[-1], 10, (255, 0, 0), -1)
    return Image.fromarray(annotated_image)

# Parse, process, and visualize the trajectory
coordinates = parse_trajectory(text_outputs)
image_width, image_height = image.size
pixel_coordinates = extract_and_denormalize_coordinates(coordinates, image_width, image_height)
if pixel_coordinates:
    final_image = draw_trajectory_with_color_transition(image, pixel_coordinates)
    final_image.save('./output_image.png')
    print("Image saved with trajectory.")
else:
    print("No coordinates found in model output.")
