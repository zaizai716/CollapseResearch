import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

texts = ["test"] * 128
inputs = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=64, truncation=True)

print(f"Shape: {inputs['input_ids'].shape}")

outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
print(f"Loss: {outputs.loss.item()}")