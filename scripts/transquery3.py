from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda")

inputs = tok("The capital of France is", return_tensors="pt").to("cuda")

cfg_dict = model.config.to_dict()
n_heads = cfg_dict['num_attention_heads']
n_kv_heads = cfg_dict['num_key_value_heads']
d_head = cfg_dict['hidden_size'] // n_heads
rope_theta = cfg_dict.get('rope_theta', 10000.0)

layer0 = model.model.layers[0]
sa = layer0.self_attn

with torch.no_grad():
    hidden = model.model.embed_tokens(inputs.input_ids)
    ln_out = layer0.input_layernorm(hidden)
    q = sa.q_proj(ln_out).float()  # [1, 5, 1536]
    k = sa.k_proj(ln_out).float()  # [1, 5, 256]
    v = sa.v_proj(ln_out).float()  # [1, 5, 256]

bsz, seq_len, _ = q.shape
q_heads = q.view(bsz, seq_len, n_heads, d_head).transpose(1, 2)      # [1, 12, 5, 128]
k_heads = k.view(bsz, seq_len, n_kv_heads, d_head).transpose(1, 2)  # [1, 2, 5, 128]
v_heads = v.view(bsz, seq_len, n_kv_heads, d_head).transpose(1, 2)  # [1, 2, 5, 128]

# Apply RoPE (non-interleaved)
half_d = d_head // 2
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_d, device="cuda").float() / half_d))
positions = torch.arange(seq_len, device="cuda").float()
angles = torch.outer(positions, inv_freq)
cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(0)
sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(0)

def apply_rope_neox(x, cos, sin):
    x1, x2 = x[..., :half_d], x[..., half_d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

q_rope = apply_rope_neox(q_heads, cos_vals, sin_vals)
k_rope = apply_rope_neox(k_heads, cos_vals, sin_vals)

# Head 0 naive attention
q0 = q_rope[0, 0]  # [5, 128]
k0 = k_rope[0, 0]  # [5, 128] (KV head 0, maps to Q heads 0-5)
v0 = v_heads[0, 0]  # [5, 128]

# QK^T scores for last query
scale = 1.0 / math.sqrt(d_head)
scores = (q0[-1:] @ k0.T * scale)  # [1, 5]
print(f"  [HF NAIVE] QK^T scores (last Q, head 0): {' '.join(f'{s:.4f}' for s in scores[0].tolist())}")

# Causal softmax (last query can attend to all)
probs = torch.softmax(scores, dim=-1)
print(f"  [HF NAIVE] Softmax probs: {' '.join(f'{p:.4f}' for p in probs[0].tolist())}")

# O = P @ V
attn_out = probs @ v0  # [1, 128]
print(f"  [HF NAIVE] Attn out head0 (last tok, first 8): {' '.join(f'{x:.4f}' for x in attn_out[0, :8].tolist())}")

# V head 0, pos 0 for sanity
print(f"  [HF NAIVE] V head0 pos0 first 8: {' '.join(f'{x:.4f}' for x in v0[0, :8].tolist())}")
