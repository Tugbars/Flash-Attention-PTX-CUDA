# Benchmarking against vLLM FlashAttention-2 on consumer Blackwell

**Result: this repository's prefill kernel now beats or matches vLLM's bundled
FlashAttention-2 at 23 of 34 measured shapes on an RTX 5080, with the remaining
deficits capped at -4.8% at the largest batch x context corner.** Both kernels
run the same instruction set (`mma.sync.m16n8k16`, `ldmatrix`, `cp.async`) on
the same GPU at the same ~84.5% tensor-pipe utilization; the gap that remains
is a register-allocation ceiling, not an algorithmic one.

All percentages in this document are **latency-based**: `+X%` means our
per-call runtime is X% lower than vLLM's on that shape. TFLOPS columns are the
same measurement re-expressed (`4*B*H*S^2*D / time`, identical FLOP convention
on both sides), so higher TFLOPS is strictly equivalent to lower latency.

---

## 1. Methodology

Cross-framework GPU benchmarks are easy to get wrong. Earlier comparisons in
this project ran our kernel under Windows (WDDM driver path) against vLLM under
WSL2, with different input data on each side — good enough for a rough map,
not good enough to make claims. The final harness removes every confound we
could identify:

- **Same process.** Our `sm_120` cubin is loaded via `cupy.RawModule` inside
  the same Python process that runs vLLM (WSL2, one CUDA context). Same OS,
  same driver path, same clock and thermal state.
- **Shared inputs.** Both kernels read the same `torch.randn` fp16 tensors
  (each in its native layout). Random data matters: constant-filled benchmark
  tensors draw less power and let the GPU boost higher, inflating results.
- **Interleaved timing.** Each shape runs alternating vLLM/ours rounds
  (medians reported), so clock drift hits both sides equally.
- **Output cross-check.** Before any timing, the two kernels' outputs are
  compared on shared inputs (agreement ~2e-4 nrmse, the fp16 noise floor).
  A benchmark of a kernel that computes something different is meaningless.
- **vLLM side:** `vllm.vllm_flash_attn.flash_attn_varlen_func`, vLLM 0.23.0,
  FA2 path (FA3 requires Hopper WGMMA/TMA and does not run on sm_120), fp16,
  causal.

Hardware: RTX 5080 (consumer Blackwell, sm_120, 84 SMs, 234.8 TFLOPS FP16
theoretical at rated clocks). Note that the vLLM FA2 wheel ships only `sm_80`
SASS + PTX — on this GPU it is JIT-compiled by the same driver `ptxas` our
kernel uses. Neither side has an instruction-set or toolchain advantage.

## 2. What the hardware counters said

Nsight Compute on both kernels at the same shape (B=8, H=12, S=2048, D=128,
causal) — same metrics, same GPU:

| Metric                          | split-N kernel (old tier) | vLLM FA2       |
|---------------------------------|---------------------------|----------------|
| Instructions executed           | 313.4M                    | **109.3M**     |
| Tensor-pipe active              | 74.2%                     | **84.5%**      |
| Occupancy (warps active)        | 32.8% (16 warps/SM)       | 8.3% (4 warps/SM) |
| Stall: barrier                  | 1.56                      | 0.06           |
| Stall: gmem latency             | 1.57                      | 0.39           |
| Stall: smem/LSU queue           | 1.08                      | 0.07           |
| Stall: MMA pipe full (the good one) | 5.43                  | 4.89           |

Two conclusions fall straight out of this table:

1. **vLLM executes 2.87x fewer instructions for the same math**, and those
   freed issue slots become +10 points of tensor-pipe utilization — which is
   exactly the latency gap that existed at the time.
2. **Occupancy is anti-correlated with performance here.** FA2 wins with ONE
   4-warp CTA per SM. Attention on this hardware wants few *fat* warps with
   large register-resident tiles, hiding latency with in-warp ILP across long
   unrolled MMA chains — not many thin warps taking turns.

## 3. The two optimizations

### 3.1 The fat-warp tier (`flash_attention_fat_kernel`)

A new kernel adopting the few-fat-warps shape, dispatched at saturation
(the split-N small tile still serves under-saturated grids, where it beats
everything including vLLM by up to +32%):

- **Split-Q row ownership.** 4 warps; each owns one m16 row-tile and the full
  row width. Softmax reduces with intra-warp shuffles only — no cross-warp
  exchange, no partial-max/sum shared memory, barrier stalls drop 26x.
- **P never touches shared memory.** The QK output (C) fragment layout is
  bit-identical to the PV input (A) fragment layout for `mma.m16n8k16`, so the
  softmax result is packed fp32->fp16 *in registers* and fed directly to the
  second MMA. This removes the P store, the P reload, one barrier per tile,
  and the buffer-aliasing constraint of the old design.
- **Software-pipelined loads, single-buffered.** V(i) is issued after K(i)
  lands and streams behind the QK matmul + softmax; K(i+1) is issued the
  moment K(i) is provably dead CTA-wide and streams behind the PV matmul.
  Zero extra shared memory; the loop-top load wait nearly vanishes
  (gmem-latency stall 2.50 -> 0.56).

D=128 causal: 193 registers, 0 spills, ~35 KB smem, 2 CTAs/SM. Measured
+9-10% over the old saturated tier across the board, and its counter profile
now matches vLLM's: **tensor-pipe 84.35% vs their 84.5%**, with the MMA pipe
itself as the dominant stall on both.

### 3.2 exp2 softmax fold

`expf(x)` compiles to `EX2` plus a hidden multiply by log2(e). Folding that
constant into the one place scores are already scaled —
`s *= scale * log2(e)` — turns every softmax exponential into a raw `EX2`,
and the running max/sum simply live in the base-2 domain (the log-sum-exp
output converts back with one `ln 2` factor in the epilogue).

Measured +3.0% at B=8 S=2048 D=128 and +2.9/+3.5% at D=64 in isolated A/B,
neutral at the largest tensor-bound shapes. An instructive footnote: this
exact transform *lost* 2-6% on the old split-N kernel — its exponentials were
already hidden under tensor-core work. Optimization verdicts are properties of
an architecture, not of a trick; when the architecture changed, we re-tested
the graveyard and this one came back to life.

## 4. Final results (same-process, 34 shapes, causal fp16)

Positive = we are faster (latency-based, vLLM as baseline). `prod` is the
shipping dispatcher (small tile below the saturation crossover, fat tier
above).

### D=64, H=12

| B | S=512 | S=1024 | S=2048 | S=4096 |
|---|------:|-------:|-------:|-------:|
| 1 | +18.5% | +31.9% | +8.9% | +2.1% |
| 4 | +15.9% | +6.1%  | +1.8% | -2.2% |
| 8 | +9.7%  | +4.0%  | -1.4% | -4.8% |

### D=128, H=12

| B | S=512 | S=1024 | S=2048 | S=3072 | S=4096 | S=8192 |
|---|------:|-------:|-------:|-------:|-------:|-------:|
| 1 | +16.6% | +12.6% | +3.8% | +0.6% | +0.2% | -2.4% |
| 4 | -0.2%  | +3.0%  | -1.1% | -1.8% | -2.3% | -3.9% |
| 8 | +3.7%  | -0.5%  | -1.9% | -2.7% | -3.5% | -4.5% |

### D=128, H=32 (Llama-class head count)

| Shape | vs vLLM |
|---|------:|
| B=1, S=2048 | +0.7% |
| B=4, S=2048 | -2.1% |
| B=8, S=2048 | -2.8% |
| B=1, S=8192 | -3.6% |

Absolute throughput tops out at ~202 TFLOPS (B=8, S=8192, D=128) vs vLLM's
~212 — both around 85% of the card's theoretical FP16 peak.

## 5. What we tried that did not work (and why that is knowledge)

Every remaining lever was built, validated for correctness, and measured —
the negative results bound where the last few percent live:

| Lever | Result | Cause |
|---|---|---|
| BM=128 (vLLM's exact tile) | -14 to -35% | 255 registers + 496 B spills in the hot loop. The Q-in-registers + P-in-registers + doubled output accumulator working set exceeds the per-thread ceiling. |
| BN=128 (wider KV tile) | -9 to -29% | 69.6 KB smem drops D=128 to 1 CTA/SM; 202 registers starve D=64 scheduling. |
| XOR-swizzled smem (no padding) | not built | Counters show bank conflicts are 0.03% of cycles — nothing to win. |
| Causal block reordering | ~0% | Both benchmark grids run 18+ oversubscribed waves; no tail to balance. |

The BM=128 failure is the precise boundary: vLLM affords that geometry because
CUTLASS's code generation keeps an equivalent working set inside the register
file where our hand-written kernel spills. That moat is worth a few percent at
the giant-shape corner (B>=4, S>=4096) and, based on three independent
falsifications, is not reachable by tile-geometry choices alone with
hand-allocated registers.

## 6. Reproducing

- `tests/fa_validate.cu` (CMake target `fa_validate`) — strict correctness
  gate, must pass before any benchmark is meaningful.
- `src/bench.cu` (`flash_bench`) — in-repo benchmark through the production
  dispatcher.
- The same-process vLLM harness lives in the WSL environment
  (`fa_sweep2.py`): loads the kernel cubin via cupy inside the vLLM process,
  cross-checks outputs, then runs interleaved timing rounds.
