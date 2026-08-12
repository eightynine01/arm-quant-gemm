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

> **Two decimal places overstate the precision.** Re-running the table three
> times, most cells move under 2% — 256³ SDOT reads 275.58 / 276.54 / 274.31 —
> but the decode row is the loose one: SDOT at 1×4096×4096 gives 63.89 / 60.49 /
> 58.71, an 8.8% spread, and the figure quoted above is the top of it. The M=1
> ratio lands between 0.72× and 0.78× depending on run. **The conclusion is not
> at risk** — every reading is far below parity — but the third digit is not
> real, and the peak figures further down are quoted as a range for the same
> reason.

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

> **Measured on the per-call-spawn harness**, which is 43–62% thread creation at
> these sizes — see [Why 16 threads loses to 8](#why-16-threads-loses-to-8). The
> 16-thread rows in particular are noise-dominated: the M=1 ratio reads 0.95×
> here and 0.83× when re-run pooled, and the peak reads 1364.5 GOPS against
> ~4400 on a spin barrier. Kept as measured, with the pooled re-run
> [below](#re-checking-the-headline-claim-on-the-fixed-harness) — the finding
> holds on both and gets sharper on the fixed one.

The mechanism is structural. `SMMLA` always emits two output rows; at M=1 the second is
padding, so half its arithmetic is discarded. Widening the tile makes that worse, not
better — the 8×8 tile spans four row pairs, so at M=1 three of four are padding. It
still beats the 4×4 kernel at M=1 (46.2 vs 38.4) because load amortisation outweighs the
extra waste, but it cannot beat an instruction that wastes nothing.

M=1 is the LLM decode step: every token after the prompt is a matrix-vector product.
Prefill runs once per prompt; decode runs once per token.

Peak on this harness: **1364.5 GOPS** (SMMLA 8×8, 8 threads, M=8), with both
kernels flattening past 8 threads and regressing at 16.

**Both numbers in that sentence were artifacts, and the explanation that used to
follow it — "bandwidth, not issue rate" — was wrong.** The table above creates
fresh OS threads on every call, which turns out to be 43–62% of the wall clock
at this size. Chased down in [Why 16 threads loses to 8](#why-16-threads-loses-to-8);
on a persistent pool the peak is **2308 GOPS** and the regression does not exist.

The table is kept as measured rather than quietly re-run, because the gap
between it and the pooled numbers is the point.

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

## How much of the machine is this actually using?

An absolute GOPS number says nothing about whether it is good. `src/roofline.rs`
measures both ceilings, and the answer changed the story twice before it settled.

### Peak throughput: SMMLA and SDOT are the same

Back-to-back instructions on distinct registers, no memory traffic, 32 per loop
iteration so the loop control does not distort the rate:

| | peak GOPS | instructions/s | per instruction |
|---|---:|---:|---:|
| SMMLA | 424 | 6.6 G/s | 64 ops |
| SDOT | 433 | 13.5 G/s | 32 ops |

**`SMMLA` does twice the work per instruction and issues at half the rate, so
peak throughput is a wash — 424 against 433 GOPS.** This is the finding the rest
of the project sits on. It explains why the tiled-vs-tiled comparison lands at
1.1–1.5× rather than the 2× the instruction description suggests: the win comes
from needing half as many instructions and half as many loads for the same work,
not from more arithmetic per second.

It also explains the M=1 loss cleanly. With no throughput advantage to spend,
the wasted half of every padded 2×2 tile has nothing to offset it.

### Which ceiling binds

| footprint | single-core load GB/s |
|---:|---:|
| 64 KiB | 154.1 |
| 4 MiB | 91.7 |
| 16 MiB (= B panel at 8×4096×4096) | 77.5 |
| 256 MiB | 54.7 |

At `m = 8` with an 8-row tile there is one row block, so A is read once (32 KiB)
and B once (16 MiB): intensity is `2·mt` = **16 ops/byte**, giving a memory
ceiling of `77.5 × 16 = 1240 GOPS`. (When both dimensions are blocked the general
form is `2·mt·nt/(mt+nt)` = 8 ops/byte, because A is re-read per column block.)

The issue ceiling is 424 GOPS. **That is what binds, and the kernel is at 85% of
it.** Not memory-bound — issue-bound, and close to done.

Three independent observations agree, which is the reason to believe it:

1. Throughput is **shape-independent**: 371 GOPS at 8×4096×4096 (B streamed from
   DRAM), 372 at 64×1024×1024, 371 at 1024×1024×1024 (B cache-resident). Measured
   bandwidth varies 154 → 55 GB/s across those footprints. A memory-bound kernel
   could not be flat across them.
2. Doubling loads in flight does nothing. `smmla_8x8_colrange_u2` unrolls `k` by
   two — same tile, same bytes, sixteen loads outstanding instead of eight — and
   measures **0.93–1.00×**. Bit-identical output. If memory were the limit this
   would have moved.
3. 85% of the issue ceiling is where a well-scheduled kernel should sit.

**An earlier version of this section claimed the opposite** — that the kernel was
memory-bound at 26% of a memory ceiling — because the issue ceiling was being
measured wrong and came out far too high. That claim was published and is
retracted here. The k-unroll experiment was designed to exploit it and failed,
which is what sent me back to check the denominator.

### Four ways the ceiling measurement was wrong

None of these produced an error. Each produced a plausible number.

| attempt | reported | what actually happened |
|---|---|---|
| constant operands | `inf` | loop folded away entirely |
| `black_box` on accumulators | 173% of peak | dependency forced through memory; measured latency, not issue rate |
| 16 chains, identical operands | 3524 GOPS | every chain computed the same value, CSE'd into one |
| `black_box` on operand arrays | 74 GOPS | an 8-vector array does not fit in registers, so the barrier spilled it to stack every iteration |

The fourth is why this is written in inline assembly. `black_box` is the only
portable barrier available, and applied to anything wider than one register it
forces a round trip through the stack — exactly the traffic a register-resident
benchmark must not have. There is no placement of it that keeps the chains both
independent and in registers.

Two of these were caught by arithmetic rather than by inspection, and the checks
are now in the code:

- **Per-cycle plausibility.** 3524 GOPS is unreadable, but 3524 / 64 ops / 3.5 GHz
  = **15.7 SMMLA per cycle** is obviously impossible. `roofline_demo()` refuses to
  print a roofline above 8 per cycle.
- **A ceiling below its kernel.** 173% of peak, and later 74 GOPS against a
  380 GOPS kernel. A bound the measured thing exceeds is self-refuting, and that
  is also checked.

The bandwidth loop had its own version of this, twice: a constant-filled buffer
let LLVM prove every load returned the same byte (487,803,629 GB/s), and after an
LCG fill it was still strength-reduced because every pass added the same bytes to
the same accumulators. What located that one was not magnitude but **slope** —
bandwidth came out rising with footprint, and bandwidth that improves as the
working set leaves cache is impossible.

## Why 16 threads loses to 8

The scaling table shows throughput *falling* from 8 threads to 16 on a machine
with 16 performance cores. An unexplained regression in a published table is
worth chasing, and the two obvious causes predict different things about the
spread of per-thread times: efficiency-core placement gives a few stragglers
several times slower than the rest, contention slows everyone equally.

Timing each worker separately at 8×4096×4096:

| threads | fastest | slowest | slow/fast | wall | spawn/join alone |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.13ms | 0.17ms | 1.26× | 0.28ms | 0.12ms |
| 12 | 0.08ms | 0.11ms | 1.43× | 0.30ms | 0.14ms |
| 16 | 0.05ms | 0.07ms | 1.38× | 0.32ms | 0.16ms |
| 20 | 0.04ms | 0.05ms | 1.22× | 0.42ms | 0.18ms |
| 24 | 0.03ms | 0.06ms | 1.83× | 0.42ms | 0.26ms |

**Neither.** The spread stays between 1.2× and 1.8×, nowhere near the ~3× that
efficiency-core placement would produce, so the work is landing on performance
cores. And the workers are not slowing down as threads are added — each one gets
proportionally faster, which is what good scaling looks like.

What is not scaling is everything else. The last column is the same
`std::thread::scope` with **empty bodies**, and it accounts for **43% to 62% of
the wall clock**. `gemm_smmla_mt` creates fresh OS threads on every call, so
adding threads buys less compute per thread while paying more for thread
creation. Past roughly 8–12 the second term wins.

At 8×4096×4096 the entire GEMM is ~0.3ms — the same order as spawning 16
threads. This is a small-batch decode shape, which is exactly where the
threading is supposed to help.

### The fix, and what it confirms

`Pool` creates its workers once and parks them on a channel. Same partition,
same kernel, same everything — the only difference is where the threads come
from:

| threads | per-call spawn | persistent pool | gain |
|---:|---:|---:|---:|
| 4 | 1068.7 | 1215.9 | 1.14× |
| 8 | 1269.4 | 2000.6 | 1.58× |
| 12 | 1280.5 | **2307.8** | 1.80× |
| 16 | 1054.9 | 2166.4 | 2.05× |
| 24 | 819.9 | 2095.4 | **2.56×** |

Bit-identical output at every thread count.

**Peak goes from 1280 to 2308 GOPS, and the regression is gone** — the pool
rises to 12 threads and then holds flat instead of collapsing. The gain grows
with thread count exactly as the diagnosis predicts, because that is where the
spawn cost was growing.

Worth stating plainly: this is the check that the explanation was right. A
hypothesis about a bottleneck earns belief when removing it moves the number,
and the retracted memory-bound claim earlier on this page failed precisely that
test — the k-unroll built to exploit it changed nothing, which is what sent me
back to remeasure.

2308 GOPS is 45% of `12 × 424` single-core issue ceilings, so the multi-threaded
path is now bound by something shared rather than by thread management.

### The pool was not the end of it

Measuring again after the pool fix: dispatch is *still* 40.5–41.8% of a call.
An mpsc send and recv per worker per job, against a job of ~0.11ms.

The obvious next suspect is memory, and it is wrong. Same op count, B panel
shrunk 16× — 8×4096×4096 (B = 16 MiB) gives 2379.8 GOPS and 128×1024×1024
(B = 1 MiB) gives 2307.2. **Nothing moves.** The remainder is synchronisation.

`SpinPool` replaces the channels with an atomic generation counter, and the
caller takes worker 0's share instead of idling at the barrier:

Three runs at each thread count, spin ÷ channel:

| threads | run 1 | run 2 | run 3 |
|---:|---:|---:|---:|
| 8 | 1.34× | 1.33× | 1.36× |
| 12 | 1.71× | 1.65× | 1.76× |
| 16 | 1.96× | 1.76× | 1.81× |
| 20 | **0.79×** | **0.85×** | **0.79×** |
| 24 | 0.82× | 0.75× | 0.82× |

Bit-identical output throughout. **Peak goes from the 1364.5 this page started
with to 4000–4500 GOPS** — 3992 to 4522 observed across six runs, quoted as a
range because a single figure would misrepresent a spread this wide — **and the
cliff is at 16 → 20 threads, exactly the performance-core count.** Past 16 the spinners have nowhere to sit but the
efficiency cores, and idle spinning stops being free the moment it competes
with real work. The pool type is a choice with a regime, not an upgrade.

**The cliff was first published in the wrong place, from one reading.** An
earlier version of this table put it at 12 → 16, on a single measurement where
16 threads showed spin losing 0.76×, and `SPIN_MAX_THREADS` shipped as 12 on
that basis. Five later runs at that same setting gave 1.28×–1.82×, all wins.
Near saturation the run-to-run spread is wide enough that **one sample decides
nothing** — while the 20- and 24-thread reversals repeat within 0.10× and are
the real boundary.

Two other ways this table was wrong before it was right, both worth naming:

- The first version built both pools before timing either, so spin workers
  burned cores while the channel pool was timed. It read 2.48× at 12 threads and
  1.16× at 16 — inflating the win and hiding the loss entirely. Each pool now
  lives in its own scope; a benchmark that leaves another thread pool alive is
  measuring both.
- The 20- and 24-thread rows did not exist at all. The table stopped at 16
  because 16 is the core count, which is precisely the region where the answer
  changes.

### Re-checking the headline claim on the fixed harness

The dispatch rule is the main result of this repo, and it had been measured with
both sides paying per-call thread creation — a cost now known to be 43–62% of
the wall clock. A comparison where half the time is thread management is partly
a measurement of the harness. So both sides were re-run pooled:

| M×N×K | 1 | 4 | 8 | 12 | 16 threads |
|---|---:|---:|---:|---:|---:|
| 1×4096×4096 *(decode)* | 0.75× | 0.80× | 0.75× | 0.76× | **0.83×** |
| 8×4096×4096 *(batch)* | 1.35× | 1.43× | 1.26× | 1.16× | 1.13× |

**The claim holds, and it got sharper.** On the spawn harness the M=1 ratio drifted
to 0.93× at 16 threads, which read like the penalty evaporating with enough
cores. It was thread-creation noise: pooled, the ratio stays in a 0.75–0.83 band
across every thread count. `SMMLA` loses at M=1 whatever the core count, and the
loss does not shrink with parallelism.

Decode throughput improved too — pooled `SDOT` at M=1 peaks at 364 GOPS against
226 before, so the shape that matters most for token generation gained 1.6×.

Cleaning a harness usually costs you a result. This one paid.

## Using it: one call that applies both rules

Two measured rules decide the work here, and neither follows from reading the
instruction descriptions. `Engine` holds them so a caller does not have to.

```rust
let eng = Engine::new(12);            // build the pool once, outside the hot path
unsafe { eng.gemm(m, n, k, mp, np, kp, &a, &bt, &pa, &pb, &mut c) };
```

| M×N×K | threads | picked | GOPS | vs scalar |
|---|---:|---|---:|---|
| 1×1024×1024 | 8 | SDOT / spin | 449.2 | match |
| 8×1024×1024 | 8 | SMMLA / spin | 2191.6 | match |
| 8×4096×4096 | 12 | SMMLA / spin | **4016.6** | match |
| 8×4096×4096 | 16 | SMMLA / chan | 2215.3 | match |
| 64×1024×1024 | 12 | SMMLA / spin | 3601.4 | match |

Every row is checked against the scalar reference, not against the fast path —
an API that encodes rules is only worth shipping if it still computes the right
thing.

Note what `Engine::new` does **not** do: it does not build a pool per call. That
would put back the 43–62% this repo spent its measurements removing, which is
why the pool is owned by the engine and the constructor is documented as the
expensive part. It also holds exactly *one* pool type rather than both, because
idle spinners take cores from whatever else is running — measured at ~25% when
a channel pool was timed next to a live spin pool.

## The crossover is at M=5, and it was published as M=2

`SMMLA_MIN_ROWS` shipped as 2 because `SMMLA` emits two rows, so M≥2 stops
wasting half of *the instruction*. The benchmarks only ever ran M=1 and M=8, so
nothing ever tested that. Sweeping it, two runs, N=K=4096, single thread —
SMMLA ÷ SDOT:

| M | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run 1 | 0.70 | 0.72 | 0.73 | 0.78 | **1.54** | 1.45 | 1.44 | 1.47 | 1.17 | 1.40 |
| run 2 | 0.73 | 0.71 | 0.73 | 0.76 | **1.47** | 1.45 | 1.49 | 1.48 | 1.11 | 1.50 |

**The crossover is M=5.** At M=2, 3 and 4 the old rule chose `SMMLA` and gave up
about 25% doing it.

The reason is in the `SDOT` column, which peaks at M=4 (232–246 GOPS) and
collapses at M=5 (149–157). `SDOT` tiles **4** rows, so M=4 fills it exactly and
M=5 opens a second row-block that is mostly padding. `SMMLA` climbs steadily as
its **8**-row tile fills. **The boundary is tile against tile, not instruction
against instruction** — the same conclusion as the rest of this page, reached
from the other direction. M=12 dipping back to 1.1× is the identical effect:
12 = 8 + 4, half a tile.

**Both dispatch constants in this repo were set at the value theory predicted
and neither was measured on both sides.** The spin cliff was published at 12 and
is at 20; this one was published at 2 and is at 5. In both cases the sweep
stopped at the number that was expected to matter, which is exactly where it
needed to keep going.

And one of them was still machine-specific after being corrected. 16 is *this*
machine's performance-core count, so shipping it as a constant is the same bug
one level up — right where it was measured, quietly wrong anywhere else.
`spin_threshold()` reads the count on macOS and falls back to the measured
value, which is stated as a fallback rather than treated as universal. The rule
is "up to the performance cores"; 16 is what that happened to be here.

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
