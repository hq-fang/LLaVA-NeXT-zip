from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import json

from PIL import Image
import requests
import copy
import torch
import numpy as np

import sys
import warnings

warnings.filterwarnings("ignore")

def download_and_explore_dataset():
    print("Downloading CV-Bench dataset...")
    cv_bench = load_dataset("nyu-visionx/CV-Bench")
    
    print("\nDataset structure:")
    print(cv_bench)
    
    test_data = cv_bench['test']
    types = set(test_data['type'])
    tasks = set(test_data['task'])
    
    print("\nTest split size:", len(test_data))
    print("Available types:", types)
    print("Available tasks:", tasks)
    
    print("\nSample example:")
    sample = test_data[0]
    for key, value in sample.items():
        if key != 'image':
            print(f"{key}: {value}")
            
    return cv_bench

class CVBench(Dataset):
    def __init__(self, args, tokenizer, image_processor, conv_template="qwen_1_5"):
        print("Initializing CVBench dataset...")
        self.cv_bench = load_dataset("nyu-visionx/CV-Bench")
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template = conv_template
        
        self.data = self.cv_bench['test'].shuffle(seed=42)
        
        # Add this back - it was accidentally removed
        self.choice_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7} 
        
        print(f"Dataset initialized with {len(self.data)} examples")

    def __getitem__(self, idx):
        image = self.data['image'][idx]
        question = self.data['question'][idx]
        choices = self.data['choices'][idx]
        choice_format = ", ".join(choices[:-1]) + ", or " + choices[-1]
        
        # Create conversation using template
        query = DEFAULT_IMAGE_TOKEN + f"\n{question} Choose between the following options: {choice_format}"
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Process answer
        answer = self.data['answer'][idx]
        answer = answer.replace("(", "").replace(")", "")
        answer = choices[self.choice_to_number[answer]]
        
        type_task = self.data['type'][idx] + "_" + self.data['task'][idx]
        
        return image, prompt, answer, f"cvbench_{type_task}"
        
    def __len__(self):
        return len(self.data)
        
    def collate_fn(self, batch):
        images, prompts, answers, datanames = zip(*batch)
        
        # Process images
        image_tensors = process_images(images, self.image_processor, model.config)
        image_tensors = [img.to(dtype=torch.float16, device="cuda") for img in image_tensors]
        image_sizes = [img.size for img in images]
        
        # Process text with padding
        all_input_ids = []
        max_length = 0
        
        # First pass: tokenize and find max length
        for prompt in prompts:
            tokens = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze(0)
            max_length = max(max_length, tokens.size(0))
            all_input_ids.append(tokens)
        
        # Second pass: pad to max length
        padded_input_ids = []
        for tokens in all_input_ids:
            padding_length = max_length - tokens.size(0)
            if padding_length > 0:
                padding = torch.ones(padding_length, dtype=tokens.dtype) * self.tokenizer.pad_token_id
                tokens = torch.cat([tokens, padding])
            padded_input_ids.append(tokens.unsqueeze(0))
        
        # Concatenate all padded inputs
        input_ids = torch.cat(padded_input_ids, dim=0).to("cuda")
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to("cuda")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": image_tensors,
            "image_sizes": image_sizes,
            "prompts": prompts,
            "answers": answers,
            "dataset": datanames,
        }

def evaluate_model(model, tokenizer, dataloader):
    correct = 0
    total = 0
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch["images"],
                image_sizes=batch["image_sizes"],
                do_sample=False,
                temperature=0,
                max_new_tokens=32,
            )
            
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for pred, prompt, answer, dataset_name in zip(predictions, batch["prompts"], batch["answers"], batch["dataset"]):
                pred_answer = pred[len(prompt):].strip()
                is_correct = pred_answer.lower() == answer.lower()
                correct += is_correct
                total += 1
                
                results.append({
                    'prediction': pred_answer,
                    'ground_truth': answer,
                    'correct': is_correct,
                    'dataset': dataset_name
                })
    
    accuracy = correct / total
    return accuracy, results

def main():
    # First download and explore the dataset
    print("=== Dataset Download and Exploration ===")
    cv_bench = download_and_explore_dataset()
    
    print("\n=== Model Loading ===")
    # Model configuration
    pretrained = "/data/input/jiafei/LLaVA-NeXT/checkpoints/onevision/Ariji_run_05b/checkpoint-51000"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": "sdpa",
    }
    
    global model, tokenizer  # Need global for use in collate_fn
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, **llava_model_args
    )
    model.eval()
    
    print("\n=== Dataset Preparation ===")
    args = {
        'batch_size': 8  # Removed num_data_points to evaluate full dataset
    }
    
    dataset = CVBench(args, tokenizer, image_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    print("\n=== Starting Full Dataset Evaluation ===")
    accuracy, results = evaluate_model(model, tokenizer, dataloader)
    
    # Group results by type and task
    type_task_results = {}
    type_results = {}
    task_results = {}
    
    for result in results:
        # Extract type and task from dataset name (format: cvbench_TYPE_TASK)
        _, type_name, task_name = result['dataset'].split('_')
        
        # Update type_task results
        type_task_key = f"{type_name}_{task_name}"
        if type_task_key not in type_task_results:
            type_task_results[type_task_key] = {'correct': 0, 'total': 0}
        type_task_results[type_task_key]['total'] += 1
        if result['correct']:
            type_task_results[type_task_key]['correct'] += 1
        
        # Update type results
        if type_name not in type_results:
            type_results[type_name] = {'correct': 0, 'total': 0}
        type_results[type_name]['total'] += 1
        if result['correct']:
            type_results[type_name]['correct'] += 1
        
        # Update task results
        if task_name not in task_results:
            task_results[task_name] = {'correct': 0, 'total': 0}
        task_results[task_name]['total'] += 1
        if result['correct']:
            task_results[task_name]['correct'] += 1
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    print("\nResults by Type:")
    for type_name, stats in sorted(type_results.items()):
        type_accuracy = stats['correct'] / stats['total']
        print(f"{type_name}: {type_accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nResults by Task:")
    for task_name, stats in sorted(task_results.items()):
        task_accuracy = stats['correct'] / stats['total']
        print(f"{task_name}: {task_accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nDetailed Results by Type and Task:")
    for type_task, stats in sorted(type_task_results.items()):
        detailed_accuracy = stats['correct'] / stats['total']
        print(f"{type_task}: {detailed_accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dict = {
        'timestamp': timestamp,
        'model': pretrained,
        'overall_accuracy': float(accuracy),  # Convert to float for JSON serialization
        'results_by_type': type_results,
        'results_by_task': task_results,
        'results_by_type_task': type_task_results,
        'detailed_results': results
    }
    
    filename = f'cvbench_evaluation_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    print(f"\nResults have been saved to '{filename}'")

if __name__ == "__main__":
    main()
