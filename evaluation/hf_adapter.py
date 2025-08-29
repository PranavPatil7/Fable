
"""Adapter for local HuggingFace transformers models.

Example:
    adapter = HFAdapter("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    print(adapter.generate("Hello"))

For large models you may want to install accelerate & bitsandbytes.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class HFAdapter:
    def __init__(self, model_name_or_path:str, device_map="auto", temperature=0.2, max_new_tokens=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer,
                             temperature=temperature, max_new_tokens=max_new_tokens)

    def generate(self, prompt:str)->str:
        out = self.pipe(prompt, do_sample=False)[0]["generated_text"]
        return out[len(prompt):].strip()
