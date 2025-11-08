import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class LocalLLM:
    def __init__(self, model_path: str, max_length: int = 256, temperature: float = 0.3):
        print(f"Loading LLM: {model_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(
            dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if device == "cuda":
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    **model_kwargs
                )
            except Exception as e:
                print(f"GPU load failed ({e}), falling back to CPU...")
                device = "cpu"
                model_kwargs["dtype"] = torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        if device == "cpu":
            self.model.to(device)
        self.max_length = max_length
        self.default_temperature = temperature
        print(f"LLM loaded on {device}")

    def generate_response(self, prompt: str, temperature: float = None) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(self.model.device)

            temp = temperature if temperature is not None else self.default_temperature

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response.strip()
            return response

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("Out of memory! Try reducing max_length or context.")
            else:
                print(f"Generation error: {e}")
            return "I'm sorry, I ran into a technical issue while generating a response."