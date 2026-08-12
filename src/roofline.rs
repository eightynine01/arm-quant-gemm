//! Instruction-issue ceiling for `SMMLA`, measured with no memory traffic.
//!
//! The benchmark reports absolute GOPS, which says nothing about whether that
//! is good. This finds the ceiling: a loop of back-to-back `SMMLA`s over
//! register-resident operands, with enough independent accumulators to hide
//! latency. Whatever that reaches is the issue limit; the GEMM number divided
//! by it is the fraction of the machine actually being used.

use std::arch::aarch64::*;
use std::hint::black_box;
use std::time::Instant;

/// # Safety
/// Requires FEAT_I8MM.
#[target_feature(enable = "i8mm")]
pub unsafe fn smmla_issue_ceiling(iters: u64) -> (f64, i32) {
    // black_box on the operands: with compile-time constants the whole loop
    // folds away and the timer reports `inf`. Asking for a number that cannot
    // be produced is worse than asking for nothing.
    let a = black_box(vdupq_n_s8(1));
    let b = black_box(vdupq_n_s8(1));
    // 16 accumulators: enough independent chains that latency cannot bind, and
    // still inside the 32 NEON registers with room for the operands.
    let mut acc = [vdupq_n_s32(0); 16];

    let t0 = Instant::now();
    for _ in 0..iters {
        // black_box on the operands only, once per iteration. Putting it on the
        // accumulators instead forces a dependency through memory and measures
        // a latency-bound *floor* — the first attempt did exactly that and
        // reported a "ceiling" slower than the real kernel.
        let a = black_box(a);
        let b = black_box(b);
        for slot in acc.iter_mut() {
            *slot = vmmlaq_s32(*slot, a, b);
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    // Each SMMLA produces a 2x2 int32 tile from 8-deep operands: 32 MACs,
    // counted as 64 ops to match the 2*M*N*K convention used elsewhere.
    let ops = iters as f64 * 16.0 * 64.0;

    // Consume the accumulators so the loop cannot be optimised away.
    let mut sink = 0i32;
    for slot in acc.iter() {
        sink = sink.wrapping_add(vgetq_lane_s32(*slot, 0));
    }
    (ops / secs / 1e9, sink)
}

/// Same shape for `SDOT`, so both instructions are measured against their own
/// ceiling rather than against each other's.
///
/// # Safety
/// Requires FEAT_DotProd.
#[target_feature(enable = "dotprod")]
pub unsafe fn sdot_issue_ceiling(iters: u64) -> (f64, i32) {
    let a = black_box(vdupq_n_s8(1));
    let b = black_box(vdupq_n_s8(1));
    let mut acc = [vdupq_n_s32(0); 16];

    let t0 = Instant::now();
    for _ in 0..iters {
        let a = black_box(a);
        let b = black_box(b);
        for slot in acc.iter_mut() {
            *slot = vdotq_s32(*slot, a, b);
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    // SDOT: 4 lanes x 4 products = 16 MACs = 32 ops.
    let ops = iters as f64 * 16.0 * 32.0;

    let mut sink = 0i32;
    for slot in acc.iter() {
        sink = sink.wrapping_add(vgetq_lane_s32(*slot, 0));
    }
    (ops / secs / 1e9, sink)
}
