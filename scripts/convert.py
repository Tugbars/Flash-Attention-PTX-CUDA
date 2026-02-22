#!/usr/bin/env python3
"""
convert.py — Convert HuggingFace (safetensors) or GGUF models to .forge format.

.forge is a flat binary format optimized for mmap-based loading in the CUDA
transformer engine. Weights are pre-transposed to column-major (NT) layout
and stored in FP16.

Usage:
    # From HuggingFace model directory or hub ID
    python convert.py --model meta-llama/Llama-3.2-1B --output llama-1b.forge

    # From local safetensors directory
    python convert.py --model ./my-model/ --output model.forge

    # From GGUF file
    python convert.py --gguf model-q8.gguf --output model.forge

Requirements:
    pip install safetensors numpy torch  (for safetensors path)
    pip install gguf numpy              (for GGUF path)

.forge binary layout:
    ┌─────────────────────────────────────────────────┐
    │ Header (fixed 512 bytes)                        │
    │   magic:            4 bytes  "FRGE"             │
    │   version:          4 bytes  uint32 = 1         │
    │   header_size:      4 bytes  uint32 = 512       │
    │   n_tensors:        4 bytes  uint32             │
    │   model_config:    96 bytes  ForgeModelConfig   │
    │   padding:        400 bytes  zeros              │
    ├─────────────────────────────────────────────────┤
    │ Tensor Table (n_tensors × 80 bytes each)        │
    │   name:           64 bytes  null-terminated     │
    │   offset:          8 bytes  uint64 (from SOF)   │
    │   nbytes:          8 bytes  uint64              │
    ├─────────────────────────────────────────────────┤
    │ Tensor Data (128-byte aligned, FP16)            │
    │   ... raw half-precision tensor bytes ...        │
    └─────────────────────────────────────────────────┘

Tensor order (per layer l = 0..n_layers-1):
    embed_tokens            — [vocab_size, d_model]  (FP16, NOT transposed)
    layers.{l}.ln1_weight   — [d_model]
    layers.{l}.wq           — [d_model, d_model]     (transposed: col-major)
    layers.{l}.wk           — [d_model, d_kv]        (transposed)
    layers.{l}.wv           — [d_model, d_kv]        (transposed)
    layers.{l}.wo           — [d_model, d_model]     (transposed)
    layers.{l}.ln2_weight   — [d_model]
    layers.{l}.w_gate       — [d_model, d_ffn]       (transposed)
    layers.{l}.w_up         — [d_model, d_ffn]       (transposed)
    layers.{l}.w_down       — [d_ffn, d_model]       (transposed)
    final_norm_weight       — [d_model]
    lm_head                 — [d_model, vocab_size]   (transposed)

"Transposed" means: HF stores [out, in] row-major. We store [in, out] row-major
= [out, in] column-major, which is what cuBLAS NT layout expects.
"""

import argparse
import json
import struct
import os
import sys
import numpy as np

MAGIC = b"FRGE"
VERSION = 1
HEADER_SIZE = 512
TENSOR_ENTRY_SIZE = 80  # 64 name + 8 offset + 8 nbytes
ALIGN = 128  # 128-byte alignment for tensor data


# =============================================================================
# ForgeModelConfig — matches the C struct layout (96 bytes)
# =============================================================================
def pack_model_config(cfg: dict) -> bytes:
    """Pack model config into 96 bytes matching C struct."""
    # int32 fields (10 × 4 = 40 bytes)
    buf = struct.pack("<10i",
        cfg["d_model"],
        cfg["n_heads"],
        cfg["n_kv_heads"],
        cfg["d_head"],
        cfg["d_ffn"],
        cfg["n_layers"],
        cfg["max_seq_len"],
        cfg["vocab_size"],
        cfg.get("bos_token_id", 1),
        cfg.get("eos_token_id", 2),
    )
    # float fields (2 × 4 = 8 bytes)
    buf += struct.pack("<2f",
        cfg.get("rms_norm_eps", 1e-5),
        cfg.get("rope_theta", 10000.0),
    )
    # bool/flags (4 × 1 = 4 bytes)
    buf += struct.pack("<4B",
        int(cfg.get("use_rotary", True)),
        int(cfg.get("use_swiglu", True)),
        int(cfg.get("use_rmsnorm", True)),
        int(cfg.get("tie_word_embeddings", False)),
    )
    # Pad to 96 bytes
    buf += b"\x00" * (96 - len(buf))
    assert len(buf) == 96
    return buf


# =============================================================================
# HuggingFace / Safetensors Loading
# =============================================================================
def load_safetensors_dir(model_path: str) -> tuple[dict, dict]:
    """Load all safetensors from a model directory. Returns (config, tensors)."""
    from safetensors import safe_open

    # Load config.json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path) as f:
        hf_config = json.load(f)

    # Find all safetensors files
    st_files = sorted([
        os.path.join(model_path, f) for f in os.listdir(model_path)
        if f.endswith(".safetensors")
    ])
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    # Load all tensors
    # Try torch first (handles bf16 natively), fall back to numpy
    tensors = {}
    use_torch = False
    try:
        import torch
        use_torch = True
    except ImportError:
        pass

    for path in st_files:
        if use_torch:
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    t = f.get_tensor(name)
                    # Convert to float16 numpy array (handles bf16 → fp16)
                    tensors[name] = t.to(torch.float16).numpy()
        else:
            # Pure numpy — try loading, handle bf16 error
            try:
                with safe_open(path, framework="numpy") as f:
                    for name in f.keys():
                        tensors[name] = f.get_tensor(name)
            except TypeError:
                print(f"  BF16 detected — install PyTorch for automatic conversion:")
                print(f"    pip install torch")
                sys.exit(1)
        print(f"  Loaded {path} ({len(tensors)} tensors total)")

    return hf_config, tensors


def load_from_hub(model_id: str) -> tuple[dict, dict]:
    """Download model from HuggingFace Hub, then load safetensors."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {model_id} from HuggingFace Hub...")
    local_dir = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "*.json"],
    )
    print(f"  Downloaded to {local_dir}")
    return load_safetensors_dir(local_dir)


# =============================================================================
# GGUF Loading
# =============================================================================
def load_gguf(gguf_path: str) -> tuple[dict, dict]:
    """Load a GGUF file. Returns (config, tensors) with FP32 numpy arrays."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: pip install gguf")
        sys.exit(1)

    print(f"Loading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)

    # Extract config from GGUF metadata
    meta = {}
    for field in reader.fields.values():
        if len(field.parts) > 1:
            val = field.parts[-1]
            if hasattr(val, 'tolist'):
                val = val.tolist()
                if len(val) == 1:
                    val = val[0]
            meta[field.name] = val

    # Map GGUF metadata keys to our config
    # GGUF uses keys like "llama.embedding_length", "llama.block_count", etc.
    arch = meta.get("general.architecture", "llama")
    hf_config = {
        "hidden_size": meta.get(f"{arch}.embedding_length", 4096),
        "num_attention_heads": meta.get(f"{arch}.attention.head_count", 32),
        "num_key_value_heads": meta.get(f"{arch}.attention.head_count_kv", 8),
        "num_hidden_layers": meta.get(f"{arch}.block_count", 32),
        "intermediate_size": meta.get(f"{arch}.feed_forward_length", 11008),
        "vocab_size": meta.get(f"{arch}.vocab_size",
                      meta.get("tokenizer.ggml.tokens", [None])),
        "max_position_embeddings": meta.get(f"{arch}.context_length", 2048),
        "rms_norm_eps": meta.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5),
        "rope_theta": meta.get(f"{arch}.rope.freq_base", 10000.0),
        "hidden_act": "silu",
        "model_type": arch,
    }
    if isinstance(hf_config["vocab_size"], list):
        hf_config["vocab_size"] = len(hf_config["vocab_size"])

    # Load tensors — dequantize to float32
    tensors = {}
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data.copy()

        # GGUF tensor types: F32=0, F16=1, Q4_0=2, Q4_1=3, Q8_0=8, etc.
        # The gguf library returns dequantized float32 for quantized types
        if data.dtype == np.float16:
            data = data.astype(np.float32)

        # Reshape from flat to proper shape
        shape = tuple(tensor.shape)
        if shape:
            data = data.reshape(shape)

        tensors[name] = data
        
    print(f"  Loaded {len(tensors)} tensors from GGUF")

    # Map GGUF tensor names to HuggingFace-style names
    mapped = {}
    gguf_to_hf = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    for name, data in tensors.items():
        if name in gguf_to_hf:
            mapped[gguf_to_hf[name]] = data
        elif name.startswith("blk."):
            # blk.N.attn_q.weight → model.layers.N.self_attn.q_proj.weight
            parts = name.split(".")
            layer_n = parts[1]
            rest = ".".join(parts[2:])
            layer_map = {
                "attn_q.weight": f"model.layers.{layer_n}.self_attn.q_proj.weight",
                "attn_k.weight": f"model.layers.{layer_n}.self_attn.k_proj.weight",
                "attn_v.weight": f"model.layers.{layer_n}.self_attn.v_proj.weight",
                "attn_output.weight": f"model.layers.{layer_n}.self_attn.o_proj.weight",
                "ffn_gate.weight": f"model.layers.{layer_n}.mlp.gate_proj.weight",
                "ffn_up.weight": f"model.layers.{layer_n}.mlp.up_proj.weight",
                "ffn_down.weight": f"model.layers.{layer_n}.mlp.down_proj.weight",
                "attn_norm.weight": f"model.layers.{layer_n}.input_layernorm.weight",
                "ffn_norm.weight": f"model.layers.{layer_n}.post_attention_layernorm.weight",
            }
            if rest in layer_map:
                mapped[layer_map[rest]] = data
            else:
                print(f"  WARNING: unmapped GGUF tensor: {name}")
                mapped[name] = data
        else:
            mapped[name] = data

    return hf_config, mapped


# =============================================================================
# Conversion Logic
# =============================================================================
def hf_config_to_forge(hf_config: dict) -> dict:
    """Convert HuggingFace config.json to forge config dict."""
    d_model = hf_config["hidden_size"]
    n_heads = hf_config["num_attention_heads"]
    n_kv_heads = hf_config.get("num_key_value_heads", n_heads)
    d_head = hf_config.get("head_dim", d_model // n_heads)

    # Llama uses SwiGLU: gate_proj + up_proj, with intermediate_size
    # being the actual FFN width (not 2/3 of it)
    d_ffn = hf_config.get("intermediate_size", d_model * 4)
    hidden_act = hf_config.get("hidden_act", "silu")
    use_swiglu = hidden_act == "silu"  # Llama's "silu" means SwiGLU FFN

    # Handle eos_token_id that might be a list
    eos = hf_config.get("eos_token_id", 2)
    if isinstance(eos, list):
        eos = eos[0]

    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "d_ffn": d_ffn,
        "n_layers": hf_config["num_hidden_layers"],
        "max_seq_len": min(hf_config.get("max_position_embeddings", 2048), 8192),
        "vocab_size": hf_config["vocab_size"],
        "bos_token_id": hf_config.get("bos_token_id", 1),
        "eos_token_id": eos,
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-5),
        "rope_theta": hf_config.get("rope_theta", 10000.0),
        "use_rotary": True,  # All modern models use RoPE
        "use_swiglu": use_swiglu,
        "use_rmsnorm": True,  # Llama, Mistral, Qwen all use RMSNorm
        "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
    }


def transpose_for_nt(tensor: np.ndarray) -> np.ndarray:
    """Transpose a 2D weight matrix for cuBLAS NT layout.
    
    HF stores weights as [out_features, in_features] row-major.
    cuBLAS NT expects B as [out_features, in_features] col-major
    = [in_features, out_features] row-major.
    
    So we just transpose: [out, in] → [in, out].
    """
    if tensor.ndim == 2:
        return np.ascontiguousarray(tensor.T)
    return tensor


def to_fp16(tensor: np.ndarray) -> np.ndarray:
    """Convert tensor to float16."""
    if tensor.dtype != np.float16:
        return tensor.astype(np.float16)
    return tensor


def align_offset(offset: int, alignment: int = ALIGN) -> int:
    """Round up to next alignment boundary."""
    return (offset + alignment - 1) & ~(alignment - 1)


def write_forge(output_path: str, forge_config: dict, hf_tensors: dict):
    """Write .forge binary file."""
    n_layers = forge_config["n_layers"]
    d_model = forge_config["d_model"]
    d_kv = forge_config["n_kv_heads"] * forge_config["d_head"]
    d_ffn = forge_config["d_ffn"]
    vocab_size = forge_config["vocab_size"]
    tie_embeddings = forge_config["tie_word_embeddings"]

    # Build ordered tensor list with forge names
    # Each entry: (forge_name, numpy_fp16_array)
    ordered = []

    def add(forge_name: str, hf_name: str, transpose: bool = False):
        if hf_name not in hf_tensors:
            print(f"  WARNING: {hf_name} not found, skipping")
            return
        t = hf_tensors[hf_name].copy()
        if transpose and t.ndim == 2:
            t = transpose_for_nt(t)
        t = to_fp16(t)
        ordered.append((forge_name, t))
        return t

    # Embedding
    add("embed_tokens", "model.embed_tokens.weight", transpose=False)

    # Per-layer weights
    for l in range(n_layers):
        pfx = f"model.layers.{l}"
        add(f"layers.{l}.ln1_weight",
            f"{pfx}.input_layernorm.weight")
        add(f"layers.{l}.wq",
            f"{pfx}.self_attn.q_proj.weight", transpose=True)
        add(f"layers.{l}.wk",
            f"{pfx}.self_attn.k_proj.weight", transpose=True)
        add(f"layers.{l}.wv",
            f"{pfx}.self_attn.v_proj.weight", transpose=True)
        add(f"layers.{l}.wo",
            f"{pfx}.self_attn.o_proj.weight", transpose=True)
        add(f"layers.{l}.ln2_weight",
            f"{pfx}.post_attention_layernorm.weight")
        add(f"layers.{l}.w_gate",
            f"{pfx}.mlp.gate_proj.weight", transpose=True)
        add(f"layers.{l}.w_up",
            f"{pfx}.mlp.up_proj.weight", transpose=True)
        add(f"layers.{l}.w_down",
            f"{pfx}.mlp.down_proj.weight", transpose=True)

    # Final norm
    add("final_norm_weight", "model.norm.weight")

    # LM head (may be tied to embeddings)
    if tie_embeddings and "lm_head.weight" not in hf_tensors:
        # Use embedding weights for lm_head
        t = hf_tensors["model.embed_tokens.weight"].copy()
        t = transpose_for_nt(t)
        t = to_fp16(t)
        ordered.append(("lm_head", t))
        print("  lm_head tied to embed_tokens (transposed copy)")
    else:
        add("lm_head", "lm_head.weight", transpose=True)

    n_tensors = len(ordered)
    print(f"\n  {n_tensors} tensors to write")

    # Compute offsets
    table_start = HEADER_SIZE
    table_size = n_tensors * TENSOR_ENTRY_SIZE
    data_start = align_offset(table_start + table_size)

    current_offset = data_start
    entries = []  # (name, offset, nbytes)
    for name, tensor in ordered:
        nbytes = tensor.nbytes
        entries.append((name, current_offset, nbytes))
        current_offset = align_offset(current_offset + nbytes)

    total_size = current_offset
    print(f"  Total file size: {total_size / 1024 / 1024:.1f} MB")

    # Write file
    with open(output_path, "wb") as f:
        # Header
        header = bytearray(HEADER_SIZE)
        header[0:4] = MAGIC
        struct.pack_into("<I", header, 4, VERSION)
        struct.pack_into("<I", header, 8, HEADER_SIZE)
        struct.pack_into("<I", header, 12, n_tensors)
        config_bytes = pack_model_config(forge_config)
        header[16:16+96] = config_bytes
        f.write(header)

        # Tensor table
        for name, offset, nbytes in entries:
            name_bytes = name.encode("utf-8")[:63]
            entry = bytearray(TENSOR_ENTRY_SIZE)
            entry[0:len(name_bytes)] = name_bytes
            struct.pack_into("<Q", entry, 64, offset)
            struct.pack_into("<Q", entry, 72, nbytes)
            f.write(entry)

        # Pad to data_start
        pos = table_start + table_size
        if pos < data_start:
            f.write(b"\x00" * (data_start - pos))

        # Tensor data
        for i, (name, tensor) in enumerate(ordered):
            expected_offset = entries[i][1]
            actual_pos = f.tell()
            assert actual_pos == expected_offset, \
                f"Offset mismatch for {name}: expected {expected_offset}, got {actual_pos}"
            f.write(tensor.tobytes())
            # Pad to alignment
            remainder = tensor.nbytes % ALIGN
            if remainder:
                f.write(b"\x00" * (ALIGN - remainder))

    print(f"  Written to {output_path}")


def print_summary(forge_config: dict, hf_tensors: dict):
    """Print model summary."""
    cfg = forge_config
    d_kv = cfg["n_kv_heads"] * cfg["d_head"]
    gqa_ratio = cfg["n_heads"] // cfg["n_kv_heads"]

    print(f"\n  Model Summary:")
    print(f"    Architecture:  Llama-style (RMSNorm + SwiGLU + RoPE + GQA)")
    print(f"    d_model:       {cfg['d_model']}")
    print(f"    n_heads:       {cfg['n_heads']} (Q) / {cfg['n_kv_heads']} (KV)  [GQA {gqa_ratio}:1]")
    print(f"    d_head:        {cfg['d_head']}")
    print(f"    d_ffn:         {cfg['d_ffn']}")
    print(f"    n_layers:      {cfg['n_layers']}")
    print(f"    vocab_size:    {cfg['vocab_size']}")
    print(f"    max_seq_len:   {cfg['max_seq_len']}")
    print(f"    rope_theta:    {cfg['rope_theta']}")

    # Estimate parameter count
    d = cfg["d_model"]
    n = cfg["n_layers"]
    V = cfg["vocab_size"]
    ffn = cfg["d_ffn"]
    # Embedding + per-layer (QKV + O + gate + up + down + 2×LN) + final_norm + lm_head
    params = V * d  # embed
    params += n * (d*d + 2*d*d_kv + d*d)  # QKV + O
    params += n * (2*d*ffn + ffn*d)  # gate + up + down
    params += n * 2 * d  # 2× LN per layer
    params += d  # final norm
    params += V * d  # lm_head
    print(f"    Parameters:    {params / 1e6:.0f}M ({params / 1e9:.2f}B)")
    print(f"    FP16 size:     {params * 2 / 1024 / 1024:.0f} MB")

    # Count what we have vs what we need
    expected_per_layer = 9  # ln1, wq, wk, wv, wo, ln2, gate, up, down
    expected_total = 1 + expected_per_layer * n + 1 + 1  # embed + layers + norm + lm_head
    found_hf = len(hf_tensors)
    print(f"    HF tensors:    {found_hf}")
    print(f"    Forge tensors: {expected_total}")


# =============================================================================
# Vocab Export — write .forge.vocab file for C tokenizer
# =============================================================================
def export_vocab(model_path: str, output_path: str):
    """Export tokenizer vocab and merges to .forge.vocab format.
    
    Reads tokenizer.json (HuggingFace format) and writes a simple text file:
      Line 1: vocab_size
      Line 2: n_merges
      Lines:  id<TAB>base64(token_bytes)
      Lines:  id1<SPACE>id2  (merge pairs)
    """
    import base64

    tok_path = os.path.join(model_path, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"  WARNING: tokenizer.json not found in {model_path}")
        print(f"  Vocab file not created. You can add it manually later.")
        return

    print(f"\n  Exporting tokenizer from {tok_path}")
    with open(tok_path, encoding="utf-8") as f:
        tok_data = json.load(f)

    model_data = tok_data.get("model", {})
    vocab = model_data.get("vocab", {})
    merges = model_data.get("merges", [])

    # Also collect added_tokens (special tokens)
    added_tokens = tok_data.get("added_tokens", [])
    for at in added_tokens:
        content = at.get("content", "")
        at_id = at.get("id", -1)
        if content and at_id >= 0:
            vocab[content] = at_id

    vocab_size = len(vocab)
    n_merges = len(merges)
    print(f"  Vocab: {vocab_size} tokens, {n_merges} merges")

    # Build the GPT-2 byte ↔ unicode mapping
    # tiktoken/Qwen tokenizers use this: each byte 0-255 is represented by a
    # specific Unicode character. We need to reverse this to get raw bytes.
    def _gpt2_bytes_to_unicode():
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    byte_to_unicode = _gpt2_bytes_to_unicode()
    unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}

    def decode_gpt2_token(token_str):
        """Decode a GPT-2/tiktoken token string to raw bytes.
        
        Each character in the token string maps to a byte via the GPT-2 mapping.
        If a character isn't in the mapping, fall back to UTF-8 encoding.
        """
        raw_bytes = bytearray()
        for ch in token_str:
            if ch in unicode_to_byte:
                raw_bytes.append(unicode_to_byte[ch])
            else:
                # Not in GPT-2 mapping — encode as UTF-8 (for CJK, special tokens etc.)
                raw_bytes.extend(ch.encode("utf-8"))
        return bytes(raw_bytes)

    # Build id->token_bytes mapping using GPT-2 decoding
    id_to_bytes = {}
    for token_str, token_id in vocab.items():
        token_bytes = decode_gpt2_token(token_str)
        id_to_bytes[token_id] = token_bytes

    # Verify: count single-byte tokens (should be close to 256 for a good BPE vocab)
    single_byte_count = sum(1 for b in id_to_bytes.values() if len(b) == 1)
    print(f"  Single-byte tokens: {single_byte_count}/256")

    # Parse merges: each merge is "tokenA tokenB" where tokenA/tokenB are
    # token strings. We need to map them to IDs.
    merge_pairs = []
    for merge_str in merges:
        # Merges are "token_a token_b" but tokens can contain spaces,
        # so split on the FIRST space only
        parts = merge_str.split(" ", 1)
        if len(parts) != 2:
            continue
        a_str, b_str = parts
        a_id = vocab.get(a_str, -1)
        b_id = vocab.get(b_str, -1)
        if a_id >= 0 and b_id >= 0:
            merge_pairs.append((a_id, b_id))

    print(f"  Valid merge pairs: {len(merge_pairs)}")

    # Write .forge.vocab
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{vocab_size}\n")
        f.write(f"{len(merge_pairs)}\n")

        # Write vocab entries sorted by ID
        for token_id in sorted(id_to_bytes.keys()):
            token_bytes = id_to_bytes[token_id]
            b64 = base64.b64encode(token_bytes).decode("ascii")
            f.write(f"{token_id}\t{b64}\n")

        # Write merge pairs
        for a_id, b_id in merge_pairs:
            f.write(f"{a_id} {b_id}\n")

    print(f"  Written to {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace or GGUF models to .forge format")
    parser.add_argument("--model", type=str,
        help="HuggingFace model ID or local directory with safetensors")
    parser.add_argument("--gguf", type=str,
        help="Path to .gguf file")
    parser.add_argument("--output", "-o", type=str, required=True,
        help="Output .forge file path")
    args = parser.parse_args()

    if not args.model and not args.gguf:
        parser.error("Specify --model (HuggingFace) or --gguf (GGUF file)")

    # Load source
    if args.gguf:
        hf_config, tensors = load_gguf(args.gguf)
    elif os.path.isdir(args.model):
        hf_config, tensors = load_safetensors_dir(args.model)
    else:
        hf_config, tensors = load_from_hub(args.model)

    # Convert config
    forge_config = hf_config_to_forge(hf_config)
    print_summary(forge_config, tensors)

    # Write .forge file
    write_forge(args.output, forge_config, tensors)

    # Export tokenizer vocab
    vocab_path = args.output + ".vocab"
    if args.gguf:
        print(f"\n  NOTE: GGUF vocab export not yet supported.")
        print(f"  Convert from HuggingFace for tokenizer support.")
    else:
        model_dir = args.model if os.path.isdir(args.model) else None
        if model_dir is None:
            # Was downloaded from hub — find the cached dir
            try:
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download(
                    args.model,
                    allow_patterns=["tokenizer.json", "tokenizer_config.json"],
                )
            except Exception:
                model_dir = None
        if model_dir:
            export_vocab(model_dir, vocab_path)

    print("\nDone. Run inference with:")
    print(f"  ./transformer_engine {args.output} \"Your prompt here\"")


if __name__ == "__main__":
    main()