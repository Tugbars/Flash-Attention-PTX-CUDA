# Benchmarking against vLLM FlashAttention-2 on consumer Blackwell

**Result: this repository's prefill kernel is faster than vLLM's bundled
FlashAttention-2 at all 34 measured shapes on an RTX 5080 (vLLM 0.34.0,
2026-11-26): by 2.8% at the tightest shape, and by 4-8% across the long-context
D=128 shapes it trailed by up to 5% before.** Both kernels run the same
instruction set (`mma.sync.m16n8k16`, `ldmatrix`, `cp.async`) on the same GPU.
The last gap was not the register-allocation ceiling an earlier version of this
document concluded it was; it was the number of load instructions per matrix
multiply (section 3.3).

History of the headline number, same 34 shapes and harness each time:

| Kernels | vLLM | Ahead | Within 2% behind | Behind by > 2% | Worst |
|---|---|---:|---:|---:|---:|
| v13 (July 2026) | 0.23.0 | 17 | 6 | 11 | -4.8% |
| restructured, pre-v14 (Nov 2026) | 0.34.0 | 21-22 | 6 | 6-7 | -5.0% |
| **v14 (Nov 2026)** | **0.34.0** | **34** | **0** | **0** | **+2.8%** |

All percentages in this document are **latency-based**: `+X%` means our
per-call runtime is X% lower than vLLM's on that shape. TFLOPS columns are the
same measurement re-expressed (`4*B*H*S^2*D / time`, identical FLOP convention
on both sides), so higher TFLOPS is strictly equivalent to lower latency. That
count is the full-attention product; causal attention executes about half of
it, so the TFLOPS figures are full-attention equivalents for comparing kernels,
not a measure of tensor-core utilization.

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
  (medians reported; 9 rounds in the current tables), so clock drift hits both
  sides equally.
- **Output cross-check.** Before any timing, the two kernels' outputs are
  compared on shared inputs (agreement ~2e-4 nrmse, the fp16 noise floor).
  A benchmark of a kernel that computes something different is meaningless.
- **vLLM side:** `vllm.vllm_flash_attn.flash_attn_varlen_func`, vLLM 0.34.0,
  FA2 path (FA3 requires Hopper WGMMA/TMA and does not run on sm_120), fp16,
  causal. The July round used vLLM 0.23.0; run through the same sweep the two
  versions are indistinguishable (median call-time difference +0.1%, under
  1.5% on every shape longer than 0.1 ms), so the version does not move any
  result here.
- **Idle GPU.** The card also drives the desktop; a round is only kept if
  nothing else is using it. A game running in the background was enough to
  turn a clean comparison into contradictory numbers.

Two caveats on reading the tables:

- **Short shapes include call overhead.** About ten of the 34 shapes run in
  under 0.1 ms. There the measurement is dominated by per-call cost — a Python
  wrapper on vLLM's side, a bare kernel launch on ours — so the large leads at
  S=512 and at B=1 overstate the kernel difference. On the 24 shapes that are
  kernel-bound the leads range from +2.8% to +14%.
- **Margins of a point or two are soft.** Run-to-run movement is about one
  point; before v14 several shapes sat at the -2% line, and the same kernels
  scored 23 "beats or matches" in July and 25 in November.

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

## 3. The optimizations

### 3.1 The fat-warp tier (`fa_prefill_splitq_kernel`, formerly `flash_attention_fat_kernel`)

A new kernel adopting the few-fat-warps shape, dispatched at saturation
(the split-N small tile still serves under-saturated grids, where it beats
everything including vLLM, by +23% to +37% in the current sweep):

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

### 3.3 Four-at-a-time operand loads (v14)

After v13 the kernel still trailed vLLM by 2-5% wherever the main loop
dominated: long context, D=128. The counters in section 2 had already named
the cause — instructions per unit of math — but the remaining instance of it
was hiding in plain sight. Every `mma.m16n8k16` took its K or V operand from
its own `ldmatrix.x2`: one load per matrix multiply, 128 loads beside 128 MMAs
per warp per tile at D=128. FlashAttention-2's CUTLASS-built loop uses the x4
form, which returns the operands of two adjacent n8 tiles at once: one load
per two MMAs.

Switching both phases to `ldmatrix.x4` (plain for K, `.trans` for V) measured,
in paired same-process A/B:

| Change | D=128 | D=64 |
|---|---:|---:|
| x4 loads for K only | +2.4 to +3.0% | +0.7% |
| x4 loads for V only | +0.6 to +1.3% | +0.9% |
| **both** | **+6.4 to +6.9%** | +0.5% |

The two halves are worth more together than apart, and the effect is confined
to D=128, where the tile is matrix-multiply-bound; at D=64 the softmax
arithmetic dominates and the loads were never the limit.

### 3.4 Mask only the tiles that need it

Every tile used to run the causal/length mask over all 64 score columns — 64
compare-and-selects per thread, interleaved with the Q K^T chains — although
only the last tile of a block (or the last two, for append-style varlen) can
contain a masked position. One CTA-uniform test per tile now skips the mask
everywhere else: +1.6 to +2.2% at both head dims.

Two smaller changes ride along. A full K/V tile is copied with one
unpredicated `cp.async` per chunk instead of the bounds-checked, clamped form
(+1.1% at D=64, neutral at D=128). And the D=64 kernels carry an occupancy
floor in their launch bounds: four blocks per SM requires at most 128
registers per thread, and the varlen instantiation had drifted to 129, which
cost 4% on small varlen shapes. The small split-N tile got the same x4 loads
(+1.9 to +3.2% at D=64, neutral at D=128).

## 4. Results (same-process, 34 shapes, causal fp16, vLLM 0.34.0)

Positive = we are faster (latency-based, vLLM as baseline), measured through
the shipping dispatch rule (small tile below the saturation crossover, fat
tier above). 2026-11-26, 9 interleaved rounds per shape.

### D=64, H=12

| B | S=512 | S=1024 | S=2048 | S=4096 |
|---|------:|-------:|-------:|-------:|
| 1 | +29.6% | +36.8% | +16.1% | +8.5% |
| 4 | +27.2% | +14.5% | +9.3%  | +4.8% |
| 8 | +24.3% | +13.8% | +8.0%  | +2.8% |

### D=128, H=12

| B | S=512 | S=1024 | S=2048 | S=3072 | S=4096 | S=8192 |
|---|------:|-------:|-------:|-------:|-------:|-------:|
| 1 | +22.8% | +24.0% | +12.0% | +9.7% | +8.4% | +5.6% |
| 4 | +17.6% | +14.1% | +9.0%  | +7.7% | +6.2% | +4.7% |
| 8 | +20.8% | +11.2% | +8.2%  | +7.0% | +5.6% | +4.3% |

### D=128, H=32 (Llama-class head count)

| Shape | vs vLLM |
|---|------:|
| B=1, S=2048 | +11.0% |
| B=4, S=2048 | +8.3% |
| B=8, S=2048 | +7.2% |
| B=1, S=8192 | +4.8% |

Absolute throughput tops out at ~218 TFLOPS (B=8, S=8192, D=128) vs vLLM's
~209, and at ~212 for D=64 (B=8, S=4096). A native Windows run through the
real dispatcher agrees on the size of the v14 step: +4-5% at D=64, +7-9% at
D=128 and +6-7% on the varlen path over the kernels it replaced.

For reference, the July tables (v13 against vLLM 0.23.0) ran from +31.9% down
to -4.8%, with every shape at B>=4, S>=3072 behind.

## 5. What we tried that did not work (and why that is knowledge)

Every remaining lever was built, validated for correctness, and measured —
the negative results bound where the last few percent live:

| Lever | Result | Cause |
|---|---|---|
| BM=128 (vLLM's exact tile) | -14 to -35% | 255 registers + 496 B spills in the hot loop. The Q-in-registers + P-in-registers + doubled output accumulator working set exceeds the per-thread ceiling. |
| BN=128 (wider KV tile) | -9 to -29% | 69.6 KB smem drops D=128 to 1 CTA/SM; 202 registers starve D=64 scheduling. |
| XOR-swizzled smem (no padding) | not built | Counters show bank conflicts are 0.03% of cycles — nothing to win. |
| Causal block reordering | ~0% | Both benchmark grids run 18+ oversubscribed waves; no tail to balance. |

The v14 round added its own, each measured alone against the then-best kernel
(D=128 unless noted):

| Lever | Result | Reading |
|---|---|---|
| BM=128 on 8 warps (per-thread state unchanged, K/V re-read half as often) | -6 to -10% | Re-read volume was never the long-context problem. |
| Start loading K(i+1) one phase earlier (extra barrier after Q K^T) | -7.2 to -7.9% | Load latency is not exposed; the extra barrier and reshuffled registers cost far more. |
| Pack P before the V barrier instead of inside the P V loop | -3.9 to -6.0% | Fewer instructions in the multiply phase, slower kernel. |
| Skip the fully-masked multiplies of the diagonal tile (two implementations) | -2.4 to -4.4% | Trims 3/8 of one tile; pushed the kernel from 189 to 201-209 registers. |
| Fold the score scale into the exponent with an FMA | -2.7 to -3.5% | (neutral at D=64 once the copy fast path was in) |
| Three blocks per SM with the 64x64 tile | -2.7 to -3.2% | Impossible anyway: 3 x 34.8 KB exceeds the SM's 102 KB; the forced register cut only adds spills. |
| k-outer / n-inner Q K^T order; rescale inside the P V loop | about -1% each | |
| Drop the redundant end-of-tile barrier; skip the rescale when no max moved; scale after the phase; four accumulators in rotation; 32-bit strides | neutral | |

Two things changed in how to read this section. First, the conclusion drawn
from the BM=128 failure in the v13 round — that the last few percent were a
register-allocation moat, unreachable for hand-allocated registers — located
the gap in the wrong place: it was the load-instruction count of section 3.3,
and it was reachable with a one-line change per phase. Second, what does hold
is the sensitivity. At D=128 the good state is 189-190 registers; variants a
dozen registers above it lost 2-4% with the block count per SM unchanged, and
several changes that removed instructions made the kernel slower. Counting
instructions predicted the big win and mispredicted most of the small ones;
every variant has to be measured.

One lever is measured but not applied: a 64x32 tile at D=128 fits three
blocks per SM and is +1.6 to +3.4% at B=8 (and at H=32 with B>=4), about +1%
at B=4, but loses at B=1 (-0.5 to -3.4%, and -19% on the smallest saturated
grid). It needs a third dispatch rule keyed on batch x heads.

## 6. Reproducing

- `tests/fa_validate.cu` (CMake target `fa_validate`) — strict correctness
  gate, must pass before any benchmark is meaningful.
- `src/bench.cu` (`flash_bench`) — in-repo benchmark through the production
  dispatcher.
- The same-process vLLM harness lives in the WSL environment
  (`fa_sweep3.py`; `fa_sweep2.py` is the July version for the pre-restructure
  kernels): loads the kernel cubin via cupy inside the vLLM process,
  cross-checks outputs, then runs interleaved timing rounds. The cubin is the
  whole `kernels/flash_attention.cu` compiled with `nvcc -cubin`; the
  restructured kernels take a single struct argument, which cupy passes by
  value from a one-element numpy structured array.
- Kernel variants were compared the same way: each variant compiled to its
  own cubin, all of them loaded into one process next to vLLM, 25 interleaved
  rounds, median of the paired ratios.
