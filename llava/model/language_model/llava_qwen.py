#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from llava.utils import rank0_print
import re

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_action_accuracy(self, outputs, inputs):
        # Convert model logits into predicted token IDs.
        # We assume outputs.logits has shape [batch_size, seq_len, vocab_size]
        predicted_ids = outputs.logits.argmax(dim=-1)  # [batch_size, seq_len]

        # Decode each prediction into a string using the trainer's tokenizer.
        # skip_special_tokens=True removes tokens like <s>, </s>, etc.
        decoded_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in predicted_ids
        ]

        # Extract predicted action tokens from the decoded text.
        # We assume the text contains a segment like "Action: [token1, token2, ...]"
        batch_action_preds = []
        for text in decoded_texts:
            match = re.search(r"Action:\s*\[(.*?)\]", text)
            if match:
                # Split tokens by comma and strip whitespace.
                preds = [tok.strip() for tok in match.group(1).split(',')]
                batch_action_preds.append(preds)
            else:
                batch_action_preds.append([])

        # Retrieve the ground truth actions from the labels tensor.
        # Here, inputs["labels"] is expected to be a tensor of shape [batch, seqlen].
        labels_tensor = inputs
        batch_action_gt = []
        if labels_tensor is not None:
            # Ensure tensor is on CPU and detached.
            # labels_tensor = labels_tensor.detach().cpu()
            # Decode each label sequence.
            for label_ids in labels_tensor:
                label_ids_filtered = [int(token) for token in label_ids if token != -100]
                decoded = self.tokenizer.decode(label_ids_filtered, skip_special_tokens=True)
                match = re.search(r"Action:\s*\[(.*?)\]", decoded)
                if match:
                    tokens = [tok.strip() for tok in match.group(1).split(',')]
                    batch_action_gt.append(tokens)
                else:
                    batch_action_gt.append([])
        else:
            # If no labels are provided, create empty ground truth for each instance.
            batch_action_gt = [[] for _ in range(len(decoded_texts))]

        # Compute accuracy by comparing the predicted and ground truth action tokens.
        total_correct = 0
        total_count = 0

        for preds, gt in zip(batch_action_preds, batch_action_gt):
            # Only compare if both prediction and ground truth lists are non-empty and of equal length.
            if len(preds) == len(gt) and len(gt) > 0:
                correct = [1 if preds[i] == gt[i] else 0 for i in range(len(gt))]
                total_correct += sum(correct)
                total_count += len(gt)

        action_accuracy = total_correct / total_count if total_count > 0 else 0.0
        return action_accuracy

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            results = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # rank0_print(f'action accuracy: {round(self.compute_action_accuracy(results, labels), 6)}')
            
            return results

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
