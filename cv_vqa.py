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

class CVBench(Dataset):
    def __init__(self, tokenizer, image_processor):
        cv_bench = load_dataset("nyu-visionx/CV-Bench")
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        self.data = cv_bench['test']
        self.choice_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

    def __getitem__(self, idx):
        image = self.data['image'][idx]
        question = self.data['question'][idx]
        choices = self.data['choices'][idx]
        choice_format = ", ".join(choices[:-1]) + ", or " + choices[-1]

        image_prompt_format = "<image>"
        prompt = f"{question} Choose between the following options: {choice_format}"

        answer = self.data['answer'][idx]
        answer = answer.replace("(", "").replace(")", "")
        answer = choices[self.choice_to_number[answer]]

        type_task = self.data['type'][idx] + "_" + self.data['task'][idx]
        
        return [image,], prompt, answer, type_task

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
    log_and_print("Loading dataset...")
    dataset = CVBench(tokenizer, image_processor)
    total_samples = len(dataset)
    log_and_print(f"Total test samples: {total_samples}")
    
    correct = 0
    total = 0
    results = {}
    detailed_results = []  # Store per-sample results

    log_and_print("\nStarting evaluation...")
    with torch.no_grad():
        for idx in range(total_samples):
            images, prompt, correct_answer, task_type = dataset[idx]
            
            # Process image
            image_tensor = process_images(images, image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            # Prepare conversation
            conv = conv_templates["qwen_1_5"].copy()
            full_prompt = DEFAULT_IMAGE_TOKEN + prompt
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # Generate response
            input_ids = tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            
            response = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[images[0].size],
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
                'is_correct': is_correct
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
    log_file = f"results/evaluation_results_{timestamp}.txt"
    
    print(f"\nStarting evaluation... Results will be saved to {log_file}")
    results = evaluate_model(pretrained, log_file)