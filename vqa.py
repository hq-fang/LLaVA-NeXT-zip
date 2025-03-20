from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import torch
import warnings
import re

warnings.filterwarnings("ignore")

# Prompt for user inputs
pretrained = '/data/input/jiafei/LLaVA-NeXT/checkpoints/onevision/Ariji_run2/checkpoint-59000'
local_image_path = input("Enter the local image path: ")
question_input = input("Enter your question about the image: ")
question = DEFAULT_IMAGE_TOKEN + question_input  # Include the image token

# Load the pretrained model
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
}
overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
llava_model_args["overwrite_config"] = overwrite_config

# Load model and tokenizer
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, **llava_model_args
)
model.eval()

# Load and preprocess the image
image = Image.open(local_image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# Define the conversation prompt
conv_template = "qwen_1_5"
conv = conv_templates[conv_template].copy()
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

# Prepare input for the model
input_ids = tokenizer_image_token(
    prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(device)
image_sizes = [image.size]

# Generate response
response = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=8192,
)

# Decode and print the response
text_output = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
print("\nModel Response:", text_output)
