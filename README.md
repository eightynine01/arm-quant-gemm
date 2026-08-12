# arm-quant-gemm

int8 GEMM kernels for AArch64, and a measurement that changes which one you should use.

`SMMLA` (Armv8.6 `FEAT_I8MM`) does twice the multiply-accumulate work per instruction
that `SDOT` does, so the usual advice is to use it wherever the CPU supports it.
On an Apple M2 Ultra that advice is right for four of the five shapes measured here
— and **wrong for the one that runs most often in LLM inference.**

Then it gets more interesting: the effect **shrinks as you add threads**, and past
about eight it inverts. That is measured below rather than assumed, and it is the
part most likely to bite someone shipping a kernel picked on a single-core benchmark.

## The result

Apple M2 Ultra (16 performance + 8 efficiency cores, `FEAT_I8MM=1`, `FEAT_DotProd=1`),
single-threaded unless stated, `-C target-cpu=native`.

Throughput, GOPS = 2·M·N·K / s:

| M×N×K | scalar | SDOT | SMMLA | **SMMLA vs SDOT** |
|---|---:|---:|---:|---:|
| 256×256×256 | 4.37 | 92.42 | **202.71** | **2.19×** |
| 512×512×512 | 3.33 | 89.30 | **167.21** | **1.87×** |
| 1024×1024×1024 | 3.40 | 71.88 | **151.00** | **2.10×** |
| 1×4096×4096 *(decode)* | 0.74 | **48.37** | 38.19 | **0.79× — slower** |
| 8×4096×4096 | 0.74 | 48.64 | **145.65** | **2.99×** |

`SMMLA` wins by 1.9–3.0× on batched shapes. At **M=1 it loses by 21%.**

## Why M=1 breaks it

`SMMLA` computes a 2×2 int32 tile from a 2×8 and an 8×2 int8 operand. It always
produces two output rows. When M=1 the second row is padding, so **half of every
instruction's arithmetic is discarded.** `SDOT` produces exactly one row and wastes
nothing, and one full-rate row beats two half-wasted ones.

This is not a tuning artifact. It is the shape of the instruction.

It matters because **M=1 is the LLM decode step.** Prefill is a big batched GEMM that
runs once per prompt; decode is a matrix-vector product that runs once per generated
token. A kernel picked for prefill throughput is the wrong kernel for the loop that
dominates wall-clock time in interactive generation.

## The part that would have been wrong to assume

An earlier draft of this README listed "does the crossover survive multi-threading?"
as an expectation rather than a measurement. Measuring it changed the conclusion.

Both kernels partitioned over output columns — columns, not rows, because at M=1
there is only one row to hand out:

| M×N×K | threads | SDOT | SMMLA | SMMLA/SDOT |
|---|---:|---:|---:|---:|
| 1×4096×4096 | 1 | 48.2 | 38.0 | **0.79×** |
| | 4 | 127.8 | 111.6 | 0.87× |
| | 8 | 169.5 | 159.8 | 0.94× |
| | 16 | 123.7 | 130.4 | **1.05×** |
| 8×4096×4096 | 1 | 47.2 | 145.0 | **3.07×** |
| | 4 | 167.4 | 464.8 | 2.78× |
| | 8 | 317.1 | **756.9** | 2.39× |
| | 16 | 433.6 | 709.7 | 1.64× |

**The M=1 penalty erodes with thread count and reverses past eight.** Both kernels
also stop scaling around 8–16 threads. Both facts point the same way: at high thread
counts these shapes are bound by memory bandwidth, not issue rate, and once you are
waiting on memory it no longer costs anything to throw away half your arithmetic.
The same effect compresses the M=8 advantage from 3.07× to 1.64×.

Peak measured: **756.9 GOPS** (SMMLA, 8 threads, M=8).

## The rule

The asymmetry makes the simple rule the right one:

```rust
pub const SMMLA_MIN_ROWS: usize = 2;

pub fn choose(m: usize) -> Kernel {
    if m >= SMMLA_MIN_ROWS { Kernel::Smmla } else { Kernel::Sdot }
}
```

| M×N×K | SDOT | SMMLA | picked | vs always-SMMLA |
|---|---:|---:|---|---:|
| 1×4096×4096 | 48.23 | 37.93 | SDOT | **1.27×** |
| 2×4096×4096 | 48.37 | 76.08 | SMMLA | 1.00× |
| 8×4096×4096 | 48.59 | 145.18 | SMMLA | 1.00× |

Choosing `SDOT` at M=1 **gives up at most ~5%** (16 threads, where the two have
converged) and **gains up to 27%** (1 thread). A thread-count-aware rule would
recover that 5%; it is not worth the extra state.

The crossover sits exactly at M=2 — the first M where `SMMLA`'s second row carries
real data.

## Correctness first

Every kernel is checked against the scalar reference before any timing is printed;
`main` exits non-zero on mismatch rather than reporting a number. Shapes are
deliberately odd — 3×5×11, 7×7×7, 17×33×65, 63×65×129 — because packed kernels
break on padding and tail handling, not on 64×64×64.

The threaded kernels are checked separately at 2, 3, 8 and 16 threads against
column counts that both do and do not divide evenly (257, 64, 129), since a
column-partitioned kernel fails by silently overlapping tiles at the seams.

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

Requires a nightly toolchain: the `SMMLA` intrinsic `vmmlaq_s32` is behind
`stdarch_neon_i8mm`. No dependencies, no data files, no network.

```sh
RUSTFLAGS="-C target-cpu=native" cargo +nightly run --release
```

Input data comes from a seeded xorshift, so results are reproducible and comparable
across machines without shipping a fixture.

On hardware without `FEAT_I8MM` the `SMMLA` path will fault — check
`sysctl hw.optional.arm.FEAT_I8MM` on macOS, or `/proc/cpuinfo` for `i8mm` on Linux.

## What this is not

Honest limits, so the numbers are read correctly:

- **The scalar column is a floor, not a fair baseline.** It is a naive triple loop
  with cache-hostile access to B. The 21–197× multiples against it bound what
  vectorisation buys; the meaningful comparison is SMMLA vs SDOT.
- **Packing is outside the timed loop.** That matches inference — weights are packed
  once and reused across every token — but would overstate the gain for a one-shot GEMM.
- **No cache blocking.** Throughput falls from 202 GOPS at 256³ to 151 at 1024³, which
  is a working-set effect a blocked kernel would recover.
- **Not compared against a tuned library.** This measures SMMLA against SDOT under
  identical conditions. It makes no claim against Accelerate, oneDNN, or llama.cpp.
- **One machine.** Every number is from a single M2 Ultra. The M=1 argument is
  structural and should hold on any `FEAT_I8MM` core, but the thread-count crossover
  depends on that machine's memory system and will move.

## Layout

```
src/kernels.rs   scalar / SDOT / SMMLA kernels, packing, threading, dispatch
src/main.rs      verification, then benchmarks
```

`gemm_smmla` accumulates a 4×4 output tile in four independent registers so the four
`SMMLA`s per k-step issue back to back instead of serialising on one accumulator.
The threaded variants hand each worker a disjoint column range, so no synchronisation
is needed inside the parallel region.

## License

MIT OR Apache-2.0.
