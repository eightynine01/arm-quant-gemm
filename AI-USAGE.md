# AI usage disclosure

Written with Claude (Anthropic) as the implementing agent, in a session directed by
the repository owner. Disclosed per hackathon transparency rules.

## What the AI wrote

- `src/kernels.rs` — all kernels (scalar, SDOT, SMMLA), the packing routines, the
  4×4 and 8×8 accumulator tiling, the column-partitioned threading, both thread
  pools (`Pool`, `SpinPool`), the `Engine` dispatch, and both dispatch constants.
- `src/roofline.rs` — the issue-ceiling and bandwidth measurements, including the
  inline assembly.
- `src/main.rs` — the verification harness and the benchmark loops.
- `README.md` — drafted from the program's actual output.

Effectively all of the code in this repository was written by the AI.

## What the human set

- The target: Armv8.6 `FEAT_I8MM` on Apple Silicon, and the decision to compare
  SMMLA against SDOT under identical conditions rather than against a tuned library.
- The standard of evidence: correctness verified against a scalar reference before
  any timing is reported, odd shapes included to exercise padding, and claims in the
  README stated no more strongly than the measurements support.
- The instruction to keep attacking the kernel rather than stop at the first
  publishable result. That produced three successive answers to the same question:
  a 21% SMMLA penalty at M=1 (both kernels naive), a 7.5x SMMLA win (only SMMLA
  tiled), and finally 1.1-1.5x with SDOT losing at M=1 (both tiled). The README
  leads with the third and keeps the first two, because the spread between them is
  the finding.
- The requirement that the losing side be optimised as carefully as the winning
  side before any ratio is published.

## Provenance of the numbers

Every figure in the README is real output from this program on an Apple M2 Ultra,
reproducible with the command in the README. Nothing was estimated, extrapolated,
or written by hand.

**Four claims were published and then retracted after re-measurement**, and the
README keeps each retraction next to its replacement rather than quietly editing:

- "memory-bound at 26% of a memory ceiling" — the issue ceiling was being measured
  far too high. It is issue-bound at 85%.
- "bandwidth, not issue rate" as the cause of the 16-thread regression — it was
  per-call thread creation, 43–62% of the wall clock.
- `SPIN_MAX_THREADS = 12`, from a single reading. The cliff is at 20.
- `SMMLA_MIN_ROWS = 2`, from the instruction's geometry. Measured, it is 5, and
  the old value made the dispatcher pick a 25%-slower kernel at M=2–4.

The last two were caught by cloning this repository fresh and running the command
the README gives — the AI's own working copy had accumulated state that hid them.
Peak throughput is quoted as a range (4000–4500 GOPS) because six runs span that,
and a single figure would claim precision the measurement does not have.
