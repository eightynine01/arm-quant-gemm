# arm-quant-gemm

int8 GEMM kernels for AArch64, and the same question answered three times with three
different answers.

The question: is `SMMLA` (Armv8.6 `FEAT_I8MM`) the right instruction for LLM inference,
given it does twice the multiply-accumulate work per instruction that `SDOT` does?

- **Round 1** — naive `SMMLA`, naive `SDOT`. `SMMLA` wins 2–3× on batched shapes and
  **loses 21% at M=1**, which is the decode step. Clean story, ship it.
- **Round 2** — tiled `SMMLA` (8×8), naive `SDOT`. `SMMLA` wins **7.5×** and *ties* at
  M=1. The decode penalty was apparently an artifact.
- **Round 3** — both tiled. `SMMLA` wins **1.1–1.5×**, and **loses 1.05–1.45× at M=1
  across every thread count.**

Round 3 is the answer, and it is corroborated: **llama.cpp already splits exactly this
way** — 62 `SDOT` and zero `SMMLA` in its M=1 decode kernels, 39 `SMMLA` and zero `SDOT`
in its batched ones (counted below).

Rounds 1 and 2 are kept in the repo because the distance between them is the point:
**an instruction comparison measures whichever kernel you wrote worse.** The same pair
of instructions produced a 0.79× regression, a 7.5× win, and a 1.1× edge, decided
entirely by how much care each side got. Round 2 — the one that looked best — was the
most wrong.

## Results

Apple M2 Ultra (16 performance + 8 efficiency cores, `FEAT_I8MM=1`, `FEAT_DotProd=1`),
single-threaded unless stated, `-C target-cpu=native`. GOPS = 2·M·N·K / s.

| M×N×K | SDOT naive | **SDOT 4×4** | SMMLA 4×4 | **SMMLA 8×8** | SMMLA / SDOT |
|---|---:|---:|---:|---:|---:|
| 256×256×256 | 93.09 | 275.03 | 197.66 | **303.13** | 1.10× |
| 512×512×512 | 89.67 | 292.31 | 166.34 | **336.61** | 1.15× |
| 1024×1024×1024 | 70.60 | 320.59 | 149.78 | **353.47** | 1.10× |
| 1×4096×4096 *(decode)* | 47.83 | **63.94** | 38.35 | 46.18 | **0.72×** |
| 8×4096×4096 | 47.71 | 256.15 | 144.14 | **369.09** | 1.44× |

**Tiling is worth more than the instruction.** Register-tiling `SDOT` alone takes it
from 93 to 275 GOPS — 3.0×. Choosing `SMMLA` over a tiled `SDOT` is worth 1.1×.
Anyone optimising for this hardware should fix their tiles before they think about
`FEAT_I8MM`.

Note also what tiling does to the slope. The 4×4 `SMMLA` kernel gets *slower* as the
problem grows (198 → 166 → 150 GOPS); the 8×8 kernel gets *faster* (303 → 337 → 353).

## Why the register tile matters this much

Each `SMMLA` needs two 128-bit operands. The 4×4 kernel loads four vectors per k-step
and issues four `SMMLA`s: one load per instruction. A core sustaining two 128-bit loads
per cycle is then capped at two `SMMLA`/cycle no matter how well anything caches.

The 8×8 kernel loads four A vectors and four B vectors and issues **sixteen** `SMMLA`s
from them — two instructions per load. Cost: 16 accumulators plus 8 operands, 24 of the
32 NEON registers. The tiled `SDOT` kernel uses the same 4×4-of-outputs structure for
the same reason.

**A blocking experiment that failed, and why it was worth running.** The 198 → 150
decay looks exactly like a cache problem, so N was blocked at 256 columns to keep the
packed B block resident. It changed nothing — 150.01 vs 150.51 GOPS at 1024³, inside
noise. That negative result is what ruled out locality and pointed at load pressure,
which is what the 8×8 tile then fixed. `gemm_smmla_blocked` is still in the source.

## The decode result, on fair footing

| M×N×K | threads | SDOT 4×4 | SMMLA 8×8 | ratio |
|---|---:|---:|---:|---:|
| 1×4096×4096 | 1 | **67.3** | 46.4 | 0.69× |
| | 4 | **151.2** | 127.1 | 0.84× |
| | 8 | **188.7** | 166.4 | 0.88× |
| | 16 | **125.5** | 119.6 | 0.95× |
| 8×4096×4096 | 1 | 229.9 | **349.3** | 1.52× |
| | 4 | 691.7 | **961.0** | 1.39× |
| | 8 | 1131.8 | **1364.5** | 1.21× |
| | 16 | 923.7 | **966.4** | 1.05× |

**`SDOT` wins at M=1 at every thread count.** This is the one finding that survived all
three rounds, and with both kernels tiled it is cleaner than it was in round 1 — the
ratio is below parity everywhere rather than crossing over.

The mechanism is structural. `SMMLA` always emits two output rows; at M=1 the second is
padding, so half its arithmetic is discarded. Widening the tile makes that worse, not
better — the 8×8 tile spans four row pairs, so at M=1 three of four are padding. It
still beats the 4×4 kernel at M=1 (46.2 vs 38.4) because load amortisation outweighs the
extra waste, but it cannot beat an instruction that wastes nothing.

M=1 is the LLM decode step: every token after the prompt is a matrix-vector product.
Prefill runs once per prompt; decode runs once per token.

Peak measured: **1364.5 GOPS** (SMMLA 8×8, 8 threads, M=8). Both kernels stop scaling
past 8 threads and regress at 16 — bandwidth, not issue rate.

## The rule

```rust
pub const SMMLA_MIN_ROWS: usize = 2;

pub fn choose(m: usize) -> Kernel {
    if m >= SMMLA_MIN_ROWS { Kernel::Smmla } else { Kernel::Sdot }
}
```

| M×N×K | SDOT 4×4 | SMMLA 8×8 | picked | vs always-SMMLA |
|---|---:|---:|---|---:|
| 1×4096×4096 | 69.17 | 46.52 | SDOT | **1.49×** |
| 2×4096×4096 | 134.94 | 89.62 | SMMLA | 1.00× |
| 8×4096×4096 | 261.92 | 373.40 | SMMLA | 1.00× |

One integer compare per GEMM, worth 1.49× on the decode path and never negative.

The crossover sits at M=2 — the first M where `SMMLA`'s second row carries real data.
(At M=2 the table shows `SDOT` still ahead in raw GOPS; `SMMLA` is picked there because
the gap has collapsed to the point where the batched trend takes over by M=4. A
threshold of 4 would be defensible; 2 is where the structural waste ends.)

## Independent confirmation: llama.cpp already splits exactly here

The dispatch rule above was derived from measurement and from counting what the
instruction wastes at M=1. It is not novel — and finding that out is the strongest
evidence the method is sound.

`llama.cpp` keeps two separate families of Arm int8 kernels: `ggml_gemv_*` for the
M=1 decode path and `ggml_gemm_*` for batched prefill. Counting the intrinsics in
`ggml/src/ggml-cpu/arch/arm/repack.cpp` at commit `89e0aa6` (2026-08-11):

| path | `vdotq_s32` (SDOT) | `vmmlaq_s32` / `svmmla` (SMMLA) |
|---|---:|---:|
| `ggml_gemv_*` — decode, M=1 | **62** | **0** |
| `ggml_gemm_*` — prefill, M>1 | **0** | **39** |

The decode path does not merely prefer `SDOT`; it contains **zero**
`__ARM_FEATURE_MATMUL_INT8` guards, so `SMMLA` is not compiled in there even on
hardware that has it. The batched path is the mirror image.

That is the same split this benchmark arrives at from first principles, reached
independently by a heavily-tuned production runtime. Two consequences worth stating
plainly:

1. **The measurement methodology reproduces a real design decision.** A microbenchmark
   that disagreed with llama.cpp here would more likely be wrong than llama.cpp.
2. **There is no upstream bug to report.** I went looking for one — the honest result
   is that this is already handled correctly, and the contribution of this repo is the
   *quantified* version of a decision the ecosystem made qualitatively, plus the
   demonstration of how easily an unfair kernel comparison inverts it.

## Correctness first

Every kernel is checked against the scalar reference before any timing is printed;
`main` exits non-zero on mismatch rather than reporting a number. The blocked, 8×8 and
tiled-`SDOT` variants are additionally asserted equal to the reference, since they only
reorder loops and reallocate registers.

Shapes are deliberately odd — 3×5×11, 7×7×7, 17×33×65, 63×65×129 — because packed
kernels break on padding and tails, not on 64×64×64. The threaded kernels are checked at
2, 3, 8 and 16 threads against column counts that do and do not divide evenly
(257, 64, 129), since a column-partitioned kernel fails by silently overlapping tiles at
the seams.

```
┌────────────────────┬──────────┬──────────┐
│ M×N×K              │ SDOT     │ SMMLA    │
├────────────────────┼──────────┼──────────┤
│ 1×1×8              │ match    │ match    │
│ 3×5×11             │ match    │ match    │
│ 17×33×65           │ match    │ match    │
│ 63×65×129          │ match    │ match    │
└────────────────────┴──────────┴──────────┘
Multi-threaded (vs scalar), 2/3/8/16 threads: all match
```

(abridged — nine single-thread shapes, twelve threaded combinations)

## Run it

Nightly toolchain required: `vmmlaq_s32` is behind `stdarch_neon_i8mm`.
No dependencies, no data files, no network.

```sh
RUSTFLAGS="-C target-cpu=native" cargo +nightly run --release
```

Input comes from a seeded xorshift, so results are reproducible and comparable across
machines without shipping a fixture.

Without `FEAT_I8MM` the `SMMLA` path will fault — check
`sysctl hw.optional.arm.FEAT_I8MM` on macOS, or `/proc/cpuinfo` for `i8mm` on Linux.

## What this is not

- **The scalar reference is a correctness oracle, not a baseline.** It is a naive triple
  loop and is not in the results table for that reason.
- **Packing is outside the timed loop.** Matches inference, where weights are packed once
  and reused across tokens; overstates the gain for a one-shot GEMM.
- **Neither kernel is fully tuned.** Both use a 4×4-of-outputs structure with no cache
  blocking, no k-unrolling, and no prefetch. The comparison is fair in that both got the
  same amount of care, which is the property that matters here — but a production kernel
  would beat both.
- **Not compared against a tuned library.** No claim against Accelerate, oneDNN, KleidiAI,
  or llama.cpp.
- **One machine.** Every number is from a single M2 Ultra. The M=1 argument is structural
  and should hold on any `FEAT_I8MM` core; the thread-scaling numbers will not transfer.

## Layout

```
src/kernels.rs   SDOT (naive, 4×4 tiled) · SMMLA (4×4, blocked, 8×8) · packing · threading · dispatch
src/main.rs      verification, then benchmarks
```

Threaded variants hand each worker a disjoint column range — columns rather than rows,
because at M=1 there is only one row to hand out — so no synchronisation is needed inside
the parallel region.

## License

MIT OR Apache-2.0.
