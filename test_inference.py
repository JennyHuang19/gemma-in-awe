"""Smoke test: load a Gemma model in bf16 and generate a short reply.

Usage: python test_inference.py [MODEL_ID]   (default: google/gemma-3-27b-it)
"""
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-27b-it"
print(f"Model: {MODEL_ID}")

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto"
).eval()
print(f"Loaded in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Describe a moment that might inspire a sense of awe, in three sentences."}
    ]}
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
new_tokens = out[0][inputs["input_ids"].shape[-1]:]
elapsed = time.time() - t0

print("-" * 60)
print(processor.decode(new_tokens, skip_special_tokens=True))
print("-" * 60)
print(f"{len(new_tokens)} tokens in {elapsed:.1f}s ({len(new_tokens) / elapsed:.1f} tok/s)")
