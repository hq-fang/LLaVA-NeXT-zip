cd /data/input/jiafei/GroundedVLA/LLaVA-NeXT &&
pip install --upgrade pip &&
pip install -e ".[train]" &&
pip install pynvml==11.5.0 &&
pip install accelerate==0.29.3 &&
pip install flash-attn==2.5.7 &&
pip install debugpy

import transformers

from llava.model.language_model.llava_qwen import LlavaQwenConfig
llava_cfg = LlavaQwenConfig.from_pretrained(model_path)

tokenizer = transformers.AutoTokenizer.from_pretrained('jaslee20/llava-epoch3-qwen2-unified', model_max_length=4096, padding_side="right")

decoded_texts = []
for label_seq in labels:  # assuming labels is a tensor of shape (batch_size, seq_length)
    # Filter out IGNORE_INDEX (-100) tokens
    valid_ids = [int(token) for token in label_seq if token != -100]
    # Decode the list of token IDs into text
    decoded_text = tokenizer.decode(valid_ids, skip_special_tokens=False)
    decoded_texts.append(decoded_text)

import numpy as np

decoded_texts = []
for token_seq in inputs['input_ids']:
    # Extract valid tokens (filter out -100)
    valid_ids = [int(token.item()) for token in token_seq if token.item() != -100]
    # Optionally, check that each id is within a reasonable range:
    for tid in valid_ids:
        if tid < 0 or tid >= tokenizer.vocab_size:
            print("Warning: token id out of expected range:", tid)
            valid_ids.remove(tid)
    # Convert to a numpy array of type int32, then to a list.
    valid_ids = np.array(valid_ids, dtype=np.int32).tolist()
    decoded_text = tokenizer.decode(valid_ids, skip_special_tokens=False)
    decoded_texts.append(decoded_text)
print(decoded_texts)



logits = results.logits  # shape: (batch_size, seq_length, vocab_size)
predicted_token_ids = logits.argmax(dim=-1)  # shape: (batch_size, seq_length)
predicted_texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
predicted_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
print(predicted_text)
