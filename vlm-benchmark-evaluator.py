from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import warnings
import re
from datetime import datetime
import os

warnings.filterwarnings("ignore")

class BLINK(Dataset):
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        dataset_name = 'BLINK-Benchmark/BLINK'
        SUBTASK_NAME = ['Multi-view_Reasoning', 'Relative_Depth', 'Spatial_Relation']

        self.data = []
        for subtask in SUBTASK_NAME:
            for entry in load_dataset(dataset_name, subtask)['val']:
                self.data.append((entry, subtask))

        self.choice_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

    def __getitem__(self, idx):
        entry, subtask = self.data[idx]
        question = entry['prompt'].split("?")[0] + "?"
        
        answer = entry['answer']
        answer = answer.replace("(", "").replace(")", "")
        answer = entry['choices'][self.choice_to_number[answer]]

        choice_format = ", ".join(entry['choices'][:-1]) + ", or " + entry['choices'][-1]

        images = []
        image_1 = entry['image_1']
        images.append(image_1)
        if entry['image_2'] is not None:
            image_2 = entry['image_2']
            images.append(image_2)

        prompt = f"{question} Choose between the following options: {choice_format}"
        
        return images, prompt, answer, "BLINK_" + subtask

    def __len__(self):
        return len(self.data)

def evaluate_model(pretrained_path, log_file):
    def log_and_print(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    # Model initialization
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": "sdpa",
    }
    overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
    llava_model_args["overwrite_config"] = overwrite_config

    # Log initial information
    log_and_print(f"Evaluation started at: {datetime.now()}")
    log_and_print(f"Model path: {pretrained_path}")
    log_and_print("\nLoading model...")

    # Load model and tokenizer
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained_path, None, model_name, device_map=device_map, **llava_model_args
    )
    model.eval()

    # Initialize dataset
    log_and_print("Loading BLINK dataset...")
    dataset = BLINK(tokenizer, image_processor)
    total_samples = len(dataset)
    log_and_print(f"Total test samples: {total_samples}")
    
    correct = 0
    total = 0
    results = {}
    detailed_results = []

    log_and_print("\nStarting evaluation...")
    with torch.no_grad():
        for idx in range(total_samples):
            images, prompt, correct_answer, task_type = dataset[idx]
            
            # Process multiple images
            image_tensors = [process_images([img], image_processor, model.config)[0] for img in images]
            image_tensors = [tensor.to(dtype=torch.float16, device=device) for tensor in image_tensors]

            # Prepare conversation with multiple image tokens
            conv = conv_templates["qwen_1_5"].copy()
            full_prompt = DEFAULT_IMAGE_TOKEN * len(images) + prompt
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # Generate response
            input_ids = tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            
            response = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=[img.size for img in images],
                do_sample=False,
                temperature=0,
                max_new_tokens=8192,
            )

            # Process response
            model_answer = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
            
            # Check if answer is correct
            is_correct = correct_answer.lower() in model_answer.lower()
            if is_correct:
                correct += 1
            total += 1

            # Store detailed results
            detailed_results.append({
                'sample_idx': idx,
                'task_type': task_type,
                'question': prompt,
                'correct_answer': correct_answer,
                'model_answer': model_answer,
                'is_correct': is_correct,
                'num_images': len(images)
            })

            # Update results dictionary
            if task_type not in results:
                results[task_type] = {'correct': 0, 'total': 0}
            results[task_type]['correct'] += int(is_correct)
            results[task_type]['total'] += 1

            # Print and log progress
            if (idx + 1) % 10 == 0:
                current_accuracy = (correct/total)*100
                log_and_print(f"Processed {idx + 1}/{total_samples} samples. Current accuracy: {current_accuracy:.2f}%")

    # Log final results
    log_and_print("\nFinal Results:")
    log_and_print(f"Total Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")
    log_and_print("\nResults by Task Type:")
    for task_type, scores in results.items():
        accuracy = (scores['correct'] / scores['total']) * 100
        log_and_print(f"{task_type}: {accuracy:.2f}% ({scores['correct']}/{scores['total']})")

    # Log detailed results
    log_and_print("\nDetailed Results:")
    for result in detailed_results:
        log_and_print(f"\nSample {result['sample_idx']}:")
        log_and_print(f"Task Type: {result['task_type']}")
        log_and_print(f"Number of Images: {result['num_images']}")
        log_and_print(f"Question: {result['question']}")
        log_and_print(f"Correct Answer: {result['correct_answer']}")
        log_and_print(f"Model Answer: {result['model_answer']}")
        log_and_print(f"Correct: {result['is_correct']}")

    log_and_print(f"\nEvaluation completed at: {datetime.now()}")
    return results

if __name__ == "__main__":
    pretrained = input("Enter the path to the pretrained model: ")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./blink_evaluation_results_{timestamp}.txt"
    
    print(f"\nStarting BLINK evaluation... Results will be saved to {log_file}")
    results = evaluate_model(pretrained, log_file)
