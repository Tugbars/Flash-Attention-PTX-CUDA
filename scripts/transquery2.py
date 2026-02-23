from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda")

inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
print("Token IDs:", inputs.input_ids.tolist())

# Get config via dict to avoid attribute errors
cfg_dict = model.config.to_dict()
n_heads = cfg_dict['num_attention_heads']      # 12
n_kv_heads = cfg_dict['num_key_value_heads']   # 2
d_model = cfg_dict['hidden_size']              # 1536
d_head = d_model // n_heads                     # 128
rope_theta = cfg_dict.get('rope_theta', 10000.0)

print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, d_head={d_head}, rope_theta={rope_theta}")

# Print ALL rope-related config keys
for k, v in sorted(cfg_dict.items()):
    if any(x in k.lower() for x in ['rope', 'rotary', 'theta', 'position']):
        print(f"  config.{k} = {v}")

layer0 = model.model.layers[0]
sa = layer0.self_attn

with torch.no_grad():
    hidden = model.model.embed_tokens(inputs.input_ids)
    ln_out = layer0.input_layernorm(hidden)
    q = sa.q_proj(ln_out)  # [1, 5, 1536]
    k = sa.k_proj(ln_out)  # [1, 5, 256]

print(f"\nQ+bias (last tok, first 8): {' '.join(f'{x:.4f}' for x in q[0,-1,:8].tolist())}")

bsz, seq_len, _ = q.shape
q_heads = q.view(bsz, seq_len, n_heads, d_head).transpose(1, 2).float()   # [1,12,5,128]
k_heads = k.view(bsz, seq_len, n_kv_heads, d_head).transpose(1, 2).float()

# Compute RoPE manually (non-interleaved / neox style)
# freq[i] = 1 / (theta ^ (2i / d_head)) for i in 0..half_d-1
half_d = d_head // 2
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_d, device="cuda").float() / half_d))
# For each position
positions = torch.arange(seq_len, device="cuda").float()
# angles[pos, dim] = pos * inv_freq[dim]
angles = torch.outer(positions, inv_freq)  # [seq_len, half_d]
cos_vals = torch.cos(angles)  # [seq_len, half_d]
sin_vals = torch.sin(angles)  # [seq_len, half_d]

# Non-interleaved RoPE: pair dim i with dim i+half_d
# q_rot[..., i]        = q[..., i] * cos - q[..., i+half_d] * sin
# q_rot[..., i+half_d] = q[..., i] * sin + q[..., i+half_d] * cos
def apply_rope_neox(x, cos, sin):
    # x: [batch, heads, seq, d_head]
    x1 = x[..., :half_d]       # first half
    x2 = x[..., half_d:]       # second half
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half_d]
    sin = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)

q_rope = apply_rope_neox(q_heads, cos_vals, sin_vals)
k_rope = apply_rope_neox(k_heads, cos_vals, sin_vals)

qr = q_rope[0, 0, -1]  # head 0, last token
kr = k_rope[0, 0, -1]

print(f"\n  [HF L0] Q post-RoPE head0 (last tok, dims 0-7):  {' '.join(f'{v:.4f}' for v in qr[:8].tolist())}")
print(f"  [HF L0] Q post-RoPE head0 (last tok, dims 64-71): {' '.join(f'{v:.4f}' for v in qr[64:72].tolist())}")
print(f"  [HF L0] K post-RoPE head0 (last tok, dims 0-7):  {' '.join(f'{v:.4f}' for v in kr[:8].tolist())}")

print(f"\n  cos[pos=4, dims 0-7]: {' '.join(f'{v:.6f}' for v in cos_vals[4, :8].tolist())}")
print(f"  sin[pos=4, dims 0-7]: {' '.join(f'{v:.6f}' for v in sin_vals[4, :8].tolist())}")

# Also print our cos/sin table values for dim 0
print(f"\n  inv_freq[0]={inv_freq[0]:.6f}  inv_freq[1]={inv_freq[1]:.6f}  inv_freq[63]={inv_freq[63]:.6f}")
print(f"  angle(pos=4, dim=0) = {4.0 * inv_freq[0].item():.6f}")

print(f"\n  Q pre-RoPE head0 last: dim0={q_heads[0,0,-1,0]:.4f}  dim64={q_heads[0,0,-1,64]:.4f}")
print(f"  Q post-RoPE head0 last: dim0={qr[0]:.4f}  dim64={qr[64]:.4f}")

# Verify: does HF use same or different RoPE style? Run full model
with torch.no_grad():
    out = model(**inputs)
    logits = out.logits[0, -1]
vals, idxs = logits.topk(3)
print(f"\nFull model top-3: {[(i.item(), f'{v.item():.3f}') for v,i in zip(vals, idxs)]}")