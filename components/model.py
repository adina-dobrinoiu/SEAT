import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from transformers import pipeline
from huggingface_hub import login


class LlamaModelForSequenceCompletion:
    def __init__(self, config):
        self.config = config
        self.device_map = {"": 0}
        self.base_model_name = config['base_model_name']
        self.use_local_model = config.get('use_local_model', True)

        if not self.use_local_model:
            # Login to Hugging Face using token
            hf_token = config.get('hf_token')
            if hf_token:
                login(token=hf_token)
            else:
                raise ValueError("Hugging Face token required for remote model loading.")

        if self.use_local_model:
            self.base_model_path = os.path.join(config['hf_model_folder'], self.base_model_name)
        else:
            self.base_model_path = self.base_model_name  # HF Hub ID

        if 'bitsandbytes' in config:
            self.bnb_config = BitsAndBytesConfig(**config['bitsandbytes'])
            self.dtype = self.bnb_config.bnb_4bit_compute_dtype
        else:
            self.bnb_config = None
            self.dtype = torch.float16

        # Load base model
        if self.bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=self.bnb_config,
                device_map=self.device_map,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def init_pipeline(self):
        # Create inference pipeline
        self.pipeline = pipeline(
            task="text-generation",
            max_new_tokens=256,
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=self.dtype,
            batch_size=4
        )

        # Set terminators
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
