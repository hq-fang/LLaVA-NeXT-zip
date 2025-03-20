import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    # print(model_name)
    # model_name = 'llava-epoch3-qwen2-unified-lora'
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map="cpu")

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/data/input/jiafei/GroundedVLA/checkpoint/17mar-libero-goal-lora-llava-qwen', required=False)
    parser.add_argument("--model-base", type=str, required=False, default='jaslee20/llava-epoch3-qwen2-unified')
    parser.add_argument("--save-model-path", type=str, required=False, default='/data/input/jiafei/GroundedVLA/checkpoint/17mar-libero-goal-lora-llava-qwen-merged')
    

    args = parser.parse_args()

    merge_lora(args) 