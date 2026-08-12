# AI usage disclosure

Written with Claude (Anthropic) as the implementing agent, in a session directed by
the repository owner. Disclosed per hackathon transparency rules.

## What the AI wrote

- `src/kernels.rs` — all kernels (scalar, SDOT, SMMLA), the packing routines, the
  4×4 accumulator tiling, the column-partitioned threading, and the dispatch rule.
- `src/main.rs` — the verification harness and the benchmark loops.
- `README.md` — drafted from the program's actual output.

Effectively all of the code in this repository was written by the AI.

## What the human set

- The target: Armv8.6 `FEAT_I8MM` on Apple Silicon, and the decision to compare
  SMMLA against SDOT under identical conditions rather than against a tuned library.
- The standard of evidence: correctness verified against a scalar reference before
  any timing is reported, odd shapes included to exercise padding, and claims in the
  README stated no more strongly than the measurements support.
- The instruction to measure the multi-threaded behaviour rather than assume it —
  which is what turned up the crossover reversal at 16 threads and corrected an
  earlier, overconfident version of the central claim.

## Provenance of the numbers

Every figure in the README is real output from this program on an Apple M2 Ultra,
reproducible with the command in the README. Nothing was estimated, extrapolated,
or written by hand. Where a measurement contradicted the initial hypothesis — the
M=1 advantage disappearing above eight threads — the README reports the
contradiction rather than the hypothesis.
