# arm-quant-gemm

int8 GEMM kernels for AArch64, and a result that started out wrong.

The plan was to show that `SMMLA` (Armv8.6 `FEAT_I8MM`) is the wrong instruction for
the LLM decode step. A first kernel appeared to prove it: `SMMLA` beat `SDOT` by
1.9–3.0× on batched shapes but **lost by 21% at M=1**, and M=1 is decode. Tidy story,
actionable dispatch rule, ship it.

Then the kernel got better, and the finding evaporated. With proper register tiling
`SMMLA` is **7.5× faster** on batched shapes and **ties** `SDOT` at M=1. The 21%
penalty was mostly an artifact of a kernel that was leaving 2.4× on the table.

Both results are below, because the gap between them is the actual lesson: **a
comparison between two instructions is only as good as the kernel you compare them
in.** A naive-but-reasonable implementation turned a 7.5× win into an apparent 0.79×
regression.

## Results

Apple M2 Ultra (16 performance + 8 efficiency cores, `FEAT_I8MM=1`, `FEAT_DotProd=1`),
single-threaded unless stated, `-C target-cpu=native`. GOPS = 2·M·N·K / s.

| M×N×K | scalar | SDOT | SMMLA 4×4 | **SMMLA 8×8** | 8×8 / 4×4 |
|---|---:|---:|---:|---:|---:|
| 256×256×256 | 4.26 | 92.18 | 197.97 | **303.12** | 1.53× |
| 512×512×512 | 3.33 | 90.41 | 164.09 | **335.06** | 2.04× |
| 1024×1024×1024 | 3.26 | 69.04 | 148.90 | **349.01** | 2.34× |
| 1×4096×4096 *(decode)* | 0.73 | 48.14 | 38.66 | **47.21** | 1.22× |
| 8×4096×4096 | 0.75 | 48.07 | 143.22 | **372.61** | 2.60× |

Note what the tile change did to the trend, not just the level. The 4×4 kernel gets
**slower** as the problem grows — 198 → 164 → 149 GOPS. The 8×8 kernel gets **faster**
— 303 → 335 → 349. Same arithmetic, same data, opposite slope.

## The register tile is the whole story

Each `SMMLA` needs two 128-bit operands. The 4×4 kernel loads four vectors per k-step
and issues four `SMMLA`s: **one load per instruction.** A core that sustains two
128-bit loads per cycle is then capped at two `SMMLA` per cycle regardless of how well
anything caches.

The 8×8 kernel loads four A vectors and four B vectors — eight loads — and issues
**sixteen** `SMMLA`s from them. Two instructions per load. Cost is 16 accumulators plus
8 operands = 24 of the 32 NEON registers.

That prediction was tested against the alternative explanation first:

**A blocking experiment that failed.** The 198 → 149 GOPS decay looks exactly like a
cache problem, so N was blocked at 256 columns to keep the packed B block resident.
It did nothing — 150.01 vs 150.51 GOPS at 1024³, inside run-to-run noise. The kernel
was never short of locality. Blocking is still in the source (`gemm_smmla_blocked`)
because the negative result is what ruled out the obvious diagnosis and pointed at
load pressure instead.

## What happened to the M=1 finding

| M×N×K | threads | SDOT | SMMLA 8×8 | ratio |
|---|---:|---:|---:|---:|
| 1×4096×4096 | 1 | 46.2 | 47.3 | 1.02× |
| | 4 | 123.7 | 119.2 | 0.96× |
| | 8 | 164.4 | 153.2 | 0.93× |
| | 16 | 112.3 | 114.6 | 1.02× |
| 8×4096×4096 | 1 | 48.3 | 363.9 | **7.54×** |
| | 4 | 167.6 | 990.9 | 5.91× |
| | 8 | 312.3 | **1212.6** | 3.88× |
| | 16 | 406.7 | 987.5 | 2.43× |

**At M=1 the two instructions are tied** — every ratio sits within 7% of parity, in
both directions. The clean 21% penalty measured with the 4×4 kernel is gone.

The underlying effect is real and still visible; it is just small. `SMMLA` always emits
two output rows, so at M=1 the second is padding and half its arithmetic is discarded.
Widening the tile makes that *worse*, not better: the 8×8 tile spans four row pairs, so
at M=1 three of four are padding — 75% waste against the 4×4 kernel's 50%. The 8×8
kernel still wins at M=1 (47.21 vs 38.66) because the load amortisation more than pays
for the extra waste. Two effects pulling opposite ways, and the net is a wash.

Peak measured: **1212.6 GOPS**, 8 threads, M=8. Both kernels stop scaling past 8
threads and regress at 16 — bandwidth, not issue rate.

## The dispatch rule, and why it barely matters now

```rust
pub const SMMLA_MIN_ROWS: usize = 2;

pub fn choose(m: usize) -> Kernel {
    if m >= SMMLA_MIN_ROWS { Kernel::Smmla } else { Kernel::Sdot }
}
```

| M×N×K | SDOT | SMMLA | picked | vs always-SMMLA |
|---|---:|---:|---|---:|
| 1×4096×4096 | 48.14 | 46.72 | SDOT | 1.03× |
| 2×4096×4096 | 48.18 | 89.71 | SMMLA | 1.00× |
| 8×4096×4096 | 47.62 | 373.31 | SMMLA | 1.00× |

With the 4×4 kernel this rule was worth 1.27×. With the 8×8 kernel it is worth 1.03×
and is arguably not worth the branch. It is kept because it costs one integer compare
and is never negative — but the honest summary is that **fixing the kernel mattered
roughly ten times more than choosing between the instructions.**

## Correctness first

Every kernel is checked against the scalar reference before any timing is printed;
`main` exits non-zero on mismatch rather than reporting a number. The blocked and 8×8
variants are additionally asserted equal to the 4×4 output, since they only reorder
loops and change register allocation.

Shapes are deliberately odd — 3×5×11, 7×7×7, 17×33×65, 63×65×129 — because packed
kernels break on padding and tails, not on 64×64×64. The threaded kernels are checked
at 2, 3, 8 and 16 threads against column counts that do and do not divide evenly
(257, 64, 129), since a column-partitioned kernel fails by silently overlapping tiles
at the seams.

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

- **The scalar column is a floor, not a fair baseline.** A naive triple loop with
  cache-hostile access to B. The meaningful comparison is SMMLA vs SDOT.
- **The SDOT kernel is not tiled.** It is a straightforward per-output-element dot
  product. Given what tiling did to `SMMLA`, a tiled `SDOT` would close part of the
  7.5× gap — the comparison is fair in that both are honest implementations, but it is
  not a comparison of two *equally optimised* kernels. This is the largest caveat here.
- **Packing is outside the timed loop.** Matches inference, where weights are packed
  once and reused across tokens; overstates the gain for a one-shot GEMM.
- **Not compared against a tuned library.** No claim against Accelerate, oneDNN, or
  llama.cpp.
- **One machine.** Every number is from a single M2 Ultra.

## Layout

```
src/kernels.rs   scalar / SDOT / SMMLA (4×4, blocked, 8×8), packing, threading, dispatch
src/main.rs      verification, then benchmarks
```

Threaded variants hand each worker a disjoint column range — columns rather than rows,
because at M=1 there is only one row to hand out — so no synchronisation is needed
inside the parallel region.

## License

MIT OR Apache-2.0.
