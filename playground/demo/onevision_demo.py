from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "/net/nfs/prior/jiafei/unified_VLM/LLaVA-NeXT/checkpoints/onevision/RoboPoint_Linedraw_05b_2/checkpoint-18000"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

local_image_path = "/net/nfs2.prior/jiafei/unified_VLM/datasets/slide_block_to_target/episode_37/front_0.png"

# Open the image from the local file path
image = Image.open(local_image_path)
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_logo.png?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "# Embodiment\n\nYou are a **visual assistant** specializing in planning the trajectory of a robotic agent. The agent is a **Franka robot** equipped with a single-arm end effector and a gripper used to interact with objects in its environment.\n\n# Action Space\n\nThe action space for this task is the **2D visual trace** of the end effector for the entire trajectory, which is defined by **2D coordinates (x, y)** within an image, representing the location of the robot's end effector at each timestep of its trajectory. These points indicate where the agent should move its end effector to follow the specified motion description and complete the task. The trajectory is represented as a sequential list of coordinate tuples:\n\n`[(x1, y1), (x2, y2), ...]`\n\nEach tuple (xi, yi) corresponds to the end effector's position at a specific timestep in the trajectory. The entire list of coordinates outlines the complete path that fulfills the motion description and accomplishes the task.\n\n# Observation Space\n\nYou are provided with an image captured from a **3rd Person Camera View**. This image represents the starting position of the agent before any actions have been taken.\n\n# Instruction\n\nGiven the current observation: <image>, you are tasked with performing the following action: **slide block to target**. To accomplish this, you must follow the motion description: **push against the block**. Your goal is to predict the **2D visual trace** of the end effector for the entire trajectory that satisfies the motion description and successfully completes the task.\n\nYour answer should be formatted as a sequential list of tuples:\n\n`[(x1, y1), (x2, y1), ...]`\n\nEach tuple contains the **x and y coordinates** representing the location of the end effector at each timestep of the trajectory. The coordinates should be **normalized values** between 0 and 1, corresponding to the pixel locations within the image.\n\n"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=8192,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)