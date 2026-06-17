# Copyright (c) 2024, NVIDIA CORPORATION.  Adapted from NVIDIA RULER (https://github.com/NVIDIA/RULER), Apache-2.0 License.
# Licensed under The MIT License [see LICENSE for details]
# Modefied from MInference 
import logging
from typing import Dict, List, Optional

import requests
import torch


logger = logging.getLogger(__name__)


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

        self.qwen_long_context = generation_kwargs.pop(
            "qwen_long_context",
            generation_kwargs.pop("qwen3_long_context", False),
        )
        self.qwen_long_context_max_position_embeddings = generation_kwargs.pop(
            "qwen_long_context_max_position_embeddings",
            generation_kwargs.pop("qwen3_long_context_max_position_embeddings", 131072),
        )
        self.qwen_yarn_factor = generation_kwargs.pop(
            "qwen_yarn_factor",
            generation_kwargs.pop("qwen3_yarn_factor", 4.0),
        )
        self.qwen_original_max_position_embeddings = generation_kwargs.pop(
            "qwen_original_max_position_embeddings",
            generation_kwargs.pop("qwen3_original_max_position_embeddings", 32768),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )

        if "Yarn-Llama" in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        try:
            if self.qwen_long_context and "qwen" in name_or_path.lower():
                config = AutoConfig.from_pretrained(
                    name_or_path,
                    trust_remote_code=True,
                )
                config.max_position_embeddings = (
                    self.qwen_long_context_max_position_embeddings
                )
                config.rope_scaling = {
                    "rope_type": "yarn",
                    "factor": self.qwen_yarn_factor,
                    "original_max_position_embeddings": (
                        self.qwen_original_max_position_embeddings
                    ),
                }
                self.tokenizer.model_max_length = max(
                    getattr(self.tokenizer, "model_max_length", 0),
                    self.qwen_long_context_max_position_embeddings,
                )
                logger.info(
                    "Loading Qwen long-context override for %s: max_position_embeddings=%s rope_scaling=%s",
                    name_or_path,
                    config.max_position_embeddings,
                    config.rope_scaling,
                )
                self.pipeline = None
                self.model = AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    config=config,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
            elif "llama-3" in name_or_path.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=name_or_path,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=model_kwargs,
                )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, **self.generation_kwargs)
            generated_text = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
        else:
            output = self.pipeline(
                text_inputs=prompt,
                **self.generation_kwargs,
            )
            assert len(output) == 1
            generated_text = output[0]["generated_text"]

        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}


class SeerAttnModel:
    def __init__(self, name_or_path: str, threshold, **generation_kwargs) -> None:
        from compact_attn import SeerAttnLlamaForCausalLM 

        from transformers import AutoTokenizer, pipeline, AutoConfig

        config = AutoConfig.from_pretrained(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, 
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Using threshold: ", threshold)

        model = SeerAttnLlamaForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            seerattn_sparsity_method='threshold',
            seerattn_threshold = threshold,
            use_cache=True,
            seerattn_last_block_dense=True,
        )

        self.pipeline =None
        self.model = model


        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, **self.generation_kwargs)
            generated_text = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
        else:
            output = self.pipeline(
                text_inputs=prompt,
                **self.generation_kwargs,
            )
            assert len(output) == 1
            generated_text = output[0]["generated_text"]


        # torch.cuda.empty_cache()

        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}
