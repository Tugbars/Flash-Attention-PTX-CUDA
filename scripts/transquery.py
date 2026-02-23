from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")

inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
print("Token IDs:", inputs.input_ids.tolist())

with torch.no_grad():
    out = model(**inputs)
    logits = out.logits[0, -1]

vals, idxs = logits.topk(10)
for v, i in zip(vals, idxs):
    print(f"  id={i.item():6d} logit={v.item():8.3f}  \"{tok.decode([i.item()])}\"")

# Also generate a few tokens
gen = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print("\nGreedy output:", tok.decode(gen[0]))