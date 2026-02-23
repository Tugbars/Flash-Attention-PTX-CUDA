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


import struct
with open("deepseek-r1-1.5b.forge", "rb") as f:
    hdr = f.read(512)
    # Config starts at byte 16
    ints = struct.unpack_from("<10i", hdr, 16)
    floats = struct.unpack_from("<2f", hdr, 56)
    flags = struct.unpack_from("<4B", hdr, 64)
    
    names = ["d_model","n_heads","n_kv_heads","d_head","d_ffn",
             "n_layers","max_seq","vocab","bos","eos"]
    for n, v in zip(names, ints):
        print(f"  {n:15s} = {v}")
    print(f"  rms_norm_eps    = {floats[0]}")
    print(f"  rope_theta      = {floats[1]}")
    print(f"  use_rotary      = {flags[0]}")
    print(f"  use_swiglu      = {flags[1]}")
    print(f"  use_rmsnorm     = {flags[2]}")
    print(f"  tie_embeddings  = {flags[3]}")