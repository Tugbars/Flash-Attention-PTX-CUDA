# Occupancy and the swizzle wall

*An engineering log of the v11 occupancy work — including the part that didn't work.*

This is the story of pushing the kernel's occupancy as far as it goes on an RTX 5080
(sm_120, 100 KB shared memory/SM, 65,536 registers/SM, 1536 threads/SM). Two of the
three steps shipped (they're v11). The third — the "interesting" one — is a clean,
fully-validated **negative result**, and it's worth writing down so nobody (including
future us) burns a weekend re-discovering it.

All numbers below were measured on one RTX 5080 with an **interleaved A/B harness**:
every variant is timed back-to-back in the same process, repeated for 9 rounds, and we
report the **median** TFLOPS. Interleaving makes clock/thermal drift hit all variants
equally, so the *relative* deltas are trustworthy even when the absolute numbers wander.

---

## The bottleneck: bubbles, not math

By v10 the kernel was at ~156 TFLOPS with the tensor cores only ~54% utilized. That gap
isn't wasted FLOPs — it's **bubbles**. Every KV tile, the tensor cores go idle while the
softmax runs (the `exp`, the cross-warp max/sum reductions, the `__syncthreads`). The
canonical way to hide a bubble is **occupancy**: more thread blocks resident per SM means
the scheduler can run *another* block's matrix-multiplies while this block does softmax.

Crucially, we confirmed the kernel is **occupancy/latency-bound, not compute-bound**: the
textbook softmax speedup (replace `expf` with `exp2f`, fold `log₂(e)` into the scale)
*lost* 2–6%, and skipping the causal mask on non-diagonal tiles also lost. Cutting work
that's already hidden under the tensor cores just adds scheduling noise. **Only occupancy
and load-overlap move the needle.** Keep that in mind — it's exactly why the swizzle fails.

### The occupancy ladder

| Blocks/SM | Occupancy | How we got there | Result |
|---:|---:|---|---|
| 2 | 33% | v10 baseline (37 KB smem) | — |
| 3 | 50% | **alias `smem_p` onto `smem_k`** → 28 KB | **+5 to +9%** ✅ shipped (v11) |
| 4 | 66% | XOR-swizzle → 25 KB + 64-reg cap | **−4 to −12%** ❌ this doc |

The alias win (step to 3 blocks) is documented in the README. This doc is about the
attempt to climb the last rung.

---

## The hypothesis: 4 blocks/SM

To fit a 4th block we need, simultaneously:

- **≤ 25 KB smem** — because 100 KB/SM ÷ 4 = 25 KB/block.
- **≤ 64 registers** — because 65,536 regs/SM ÷ (4 × 256 threads) = 64 regs/thread.

The register side is easy: `__launch_bounds__(256, 4)` forces the compiler to 64 regs, and
we confirmed it does so with **zero spills**. The smem side is the whole problem.

We're at 28 KB after the alias. The 3 KB to shave is exactly the `+8`-half padding on
`smem_q`/`smem_k`/`smem_v` (1 KB each). That padding isn't slack — it's what keeps
`ldmatrix` conflict-free: with a 64-half row stride (128 B = exactly 32 banks), all 8 rows
of an `ldmatrix` tile start in bank 0 → an **8-way bank conflict**. We measured naive
padding removal at **−47 to −62%**. So the padding can only go if something else provides
the conflict-avoidance. That something is an **XOR swizzle**.

---

## The swizzle

Instead of "waste a column of padding," permute the column so consecutive rows scatter
across banks. For D=64 fp16 with 16-byte (8-half) `ldmatrix` granularity, a row is exactly
8 chunks spanning 32 banks — the textbook case. The mapping, for logical row `r` and
half-column `c`:

```cuda
__device__ __forceinline__ int swz64(int row, int col) {
    return row * 64 + (col ^ ((row & 7) << 3));
}
```

The XOR flips only the **chunk bits (3–5)** of the column, never the within-chunk bits
(0–2). Two properties fall out:

- **Correctness (bijection per row).** XOR is a permutation, so it's a 1:1 remap; applied
  identically at every write and read, the kernel's output is bit-identical to the padded
  version. Within-chunk layout is untouched, so packed `__half2` pairs stay adjacent.
- **Conflict-free.** For any `ldmatrix` 8×8 tile the 8 participating rows have
  `row & 7 = 0..7`, so their chunks XOR to 8 distinct values → 8 distinct 4-bank groups.

The catch: it must be applied **identically at all 8 access sites** — the Q store + Q
`ldmatrix.x4`, the K `cp.async` + K `ldmatrix.x2.trans`, the V `cp.async` + V
`ldmatrix.x2.trans`, and the P scalar stores + P `ldmatrix.x4`. The transposed K/V loads
are the historically buggy ones (the kernel's own comments call `ldmatrix.x2.trans`
addressing "the bug that took longest to find"), and the P stores are *not* chunk-aligned
(odd within-chunk column offsets), so they need the full half-granular XOR.

### De-risking: the bit-exact oracle

A wrong swizzle silently loads the wrong data and can still *look* plausible. We neutralized
that: a swizzle changes **storage layout only**, so the output **must be bit-identical to
the padded kernel**. Any addressing mistake shows up immediately as a correctness failure
against the CPU reference. The "silent" corruption risk becomes *loud*. On top of that, a
5-agent adversarial design review independently re-derived the address math for each site
and confirmed all 8 consistent, conflict-free, with no missed sites. Both the oracle and
the review came back clean — so what follows is a genuine **performance** verdict, not a bug.

---

## Results

Three variants, all numerically identical to the v11 baseline:

- **`v_base`** — v11 (alias + staged cp.async), 78 regs, 28 KB, **3 blocks/SM**.
- **`v_swz3`** — swizzle + `launch_bounds(256,3)`, 80 regs, 25 KB, **3 blocks/SM**
  (isolates *swizzle overhead at equal occupancy*).
- **`v_swz4`** — swizzle + `launch_bounds(256,4)`, 64 regs, 25 KB, **4 blocks/SM**
  (the actual goal).

| Config | v_base (3 blk) | v_swz3 (3 blk) | v_swz4 (4 blk) |
|---|---:|---:|---:|
| B=1 H=12 S=512  | 45.6 | −1.2% | +1.1% |
| B=1 H=12 S=2048 | 106.9 | −10.8% | −14.4% |
| B=4 H=12 S=2048 | 138.8 | −2.2% | −3.2% |
| B=8 H=12 S=2048 | 153.8 | −7.2% | −8.0% |
| B=1 H=12 S=4096 | 134.1 | −6.1% | −7.9% |
| B=4 H=12 S=4096 | 159.3 | +0.1% | −7.3% |

(An earlier run let the swizzle compile uncapped — it ballooned to **124 registers** and
fell to **2 blocks/SM**, losing 5–15%. ptxas hoards address registers when unconstrained.)

---

## Why it failed

Two independent effects, and the `v_swz3` column separates them cleanly:

**1. Swizzle is strictly worse than padding *at the same occupancy*.** `v_swz3` runs at the
identical 3 blocks/SM as `v_base`, yet loses −1 to −11%. The reason is simple once you see
it: **padding's conflict-avoidance is free** — it's just a different stride constant, zero
instructions. **The swizzle's conflict-avoidance costs** — an XOR on every address plus
register pressure (the compiler wanted 124 regs; capping it forces address recompute). You
trade a free solution for a paid one and get nothing back, because…

**2. The 4th block buys nothing.** `v_swz4` (4 blocks) is no better than `v_swz3`
(3 blocks) — often slightly worse, because reaching 4 blocks forces the harsher 64-reg cap.
The softmax bubble is *already* hidden at 3 blocks/50%; the marginal occupancy from a 4th
block has almost no bubble left to hide, so it can't pay back the swizzle + register cost.

---

## The lesson

The alias win (33% → 50%) paid for two reasons that the swizzle step lacks:

1. **It was free** — pure pointer aliasing, no compute or register cost.
2. **33% was genuinely starved** — there was a lot of exposed bubble to hide.

The swizzle step (50% → 66%) has **neither**: it isn't free, and 50% already saturates the
bubble-hiding. So the rule of thumb that comes out of this:

> **Occupancy gains only pay when they're cheap *and* you're genuinely under-occupied.**
> Past that, more blocks just cost registers and instructions for a bubble that's already hidden.

For this kernel that ceiling is **3 blocks/SM, 78 registers, padded** — which is exactly
where v11 sits. The occupancy lever is exhausted.

---

## Verdict & what's left

- **Shipped (v11):** alias `smem_p`→`smem_k` + staged `cp.async`. ~170 TFLOPS at the B=8,
  S=2048 sweet spot, ~183 peak at B=4, S=4096.
- **Not shipped:** the 4-block swizzle tier. Correct and conflict-free, but a measured
  regression. All of this work lived in a scratchpad A/B harness; the production kernel was
  never touched.
- **Still open (orthogonal):** split-K / flash-decoding for under-saturated configs
  (B=1, S≥1024), which the tile dispatcher currently mis-routes to the big tile. That's a
  *grid-level* parallelism win and doesn't touch any of the occupancy machinery here.

The most valuable artifact from this experiment isn't code — it's the knowledge that the
occupancy ceiling is real and where it is, so the next optimization aims somewhere else.
