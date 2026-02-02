import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from fastapi import HTTPException

# Use it in your code
raise HTTPException(status_code=404, detail="Item not found")

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LLMService:
    def __init__(self, model_id: str, max_tokens: int, temperature: float):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"Loaded LLM: {model_id} on {DEVICE}")

    def build_prompt(self, user_query: str, ctx_docs: List[Dict]) -> str:
        context_str = "\n\n".join(
            [f"[{i+1}] {d['text']}\n(Source: {d['source']})" for i, d in enumerate(ctx_docs)]
        )
        system = (
            "You are a helpful, multilingual assistant. Use the provided context to answer factually. "
            "If the answer is not in the context, say you don't know. Keep responses concise and clear."
        )
        return f"{system}\n\nContext:\n{context_str}\n\nUser: {user_query}\nAssistant:"

    def generate(self, prompt: str) -> str:
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if "Assistant:" in full_text:
                return full_text.split("Assistant:")[-1].strip()
            return full_text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise HTTPException(status_code=500, detail="LLM generation failed")
