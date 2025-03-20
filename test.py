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
pretrained = "/data/input/jiafei/GroundedVLA/checkpoint/depthckpt"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
}
overwrite_config = {}
overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152035}
llava_model_args["overwrite_config"] = overwrite_config
#tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)  # Add any other thing you want to pass in llava_model_args

#tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
img_path = "/data/input/jiafei/datasets/Full_OXE/pose/bc_z/0001927/0075.png"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(img_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nThe task is pick up the ceramic cup. Can you predict the depth map of the image, the trajectory of the end effector and the action the robot should take?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# print(input_ids)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
# print(cont.shape)

actions = cont[0, -11:-2].unsqueeze(0)
print(actions)

text_actions = tokenizer.batch_decode(actions, skip_special_tokens=True)
predicted_actions = tokenizer.decode(actions[0], skip_special_tokens=True)
print(predicted_actions)

# text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
# predicted_text = tokenizer.decode(cont[0], skip_special_tokens=True)
# print(predicted_text)
