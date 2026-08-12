//! Both halves of the roofline: the instruction-issue ceiling and the memory
//! ceiling.
//!
//! The benchmark reports absolute GOPS, which says nothing about whether that
//! is good. The issue ceiling is a loop of back-to-back `SMMLA`s over
//! register-resident operands, with enough independent accumulators to hide
//! latency. Whatever that reaches is the issue limit.
//!
//! That alone is a misleading denominator. A kernel far below the issue ceiling
//! is not necessarily leaving anything on the table — it may be at its *memory*
//! ceiling instead, and no amount of instruction scheduling would move it.
//! Which ceiling applies is decided by arithmetic intensity, and for this GEMM
//! the intensity has a closed form.
//!
//! Take an `m x n x k` product with an `mt`-row register tile. B is re-read once
//! per block of `mt` rows, so B traffic is `(m/mt) * n * k` bytes against
//! `2*m*n*k` ops:
//!
//! ```text
//!     intensity = 2*m*n*k / ((m/mt) * n * k) = 2 * mt   ops per byte
//! ```
//!
//! The shape cancels out. The 8x8 kernel is pinned at 16 ops/byte and the 4x4
//! kernel at 8, whatever `m`, `n`, and `k` are. So the memory ceiling is just
//! `bandwidth * 2 * mt` — and the useful bandwidth is whichever level of the
//! hierarchy B actually lives in, which is why `stream_bandwidth` sweeps
//! footprints instead of reporting one number.
//!
//! Note what this rules out. At `m = 8` with an 8-row tile, B is read exactly
//! once; that is the floor, and no cache blocking can reduce it. Reaching for
//! blocking there would be optimising traffic that is already minimal.

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

/// Sustained single-core load bandwidth for a buffer of `bytes`, in GB/s.
///
/// Swept across footprints rather than reported once, because the ceiling that
/// applies to a GEMM depends on which level of the hierarchy the B panel is
/// resident in — an L2-resident B and a DRAM-resident B give the same kernel
/// two very different memory ceilings.
///
/// Eight independent accumulators, one cheap `vaddq_s8` per 16-byte load. The
/// add is there only to keep the loads live; at one ALU op per load the loop
/// cannot become arithmetic-bound, which is the failure mode that would make
/// this report a bandwidth lower than the machine's.
///
/// Two things keep the loop from being optimised out, and the first attempt
/// missed both — it filled the buffer with a constant and let the compiler see
/// the pointer, so LLVM proved every load returned the same byte and deleted
/// the whole thing. It reported 487,803,629 GB/s, and `inf` at one size.
///
/// So: the buffer is filled from an LCG, and the base pointer goes through
/// `black_box` once per pass so the contents stay opaque and the loads cannot
/// be hoisted. Same lesson as the issue-ceiling loops above — an eliminated
/// benchmark does not report zero, it reports a number too good to be true.
///
/// That still was not enough. Every pass adds the same bytes to the same
/// accumulators, so LLVM strength-reduced the whole outer loop into one pass
/// times a constant. The tell was the *shape* of the result rather than its
/// size: bandwidth came out rising with footprint, 229,446 GB/s at 64 KiB up to
/// 158,913,790 GB/s at 256 MiB. Bandwidth that improves as the working set
/// leaves cache is impossible, and reading that slope was what located the bug.
/// Passing the accumulators through `black_box` once per pass forces each pass
/// to actually happen; once per *pass* and not per load, so the cost is
/// amortised over thousands of loads instead of serialising them.
pub fn stream_bandwidth(bytes: usize, iters: u64) -> (f64, i32) {
    let mut buf = vec![0i8; bytes];
    let mut s = 0x2545F4914F6CDD1Du64;
    for slot in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *slot = (s >> 56) as i8;
    }

    let mut acc = [unsafe { vdupq_n_s8(0) }; 8];
    let step = 16 * 8;
    let limit = bytes - (bytes % step);

    let t0 = Instant::now();
    for _ in 0..iters {
        let base = black_box(buf.as_ptr());
        let mut off = 0;
        while off < limit {
            unsafe {
                let p = base.add(off);
                for (j, slot) in acc.iter_mut().enumerate() {
                    *slot = vaddq_s8(*slot, vld1q_s8(p.add(j * 16)));
                }
            }
            off += step;
        }
        acc = black_box(acc);
    }
    let secs = t0.elapsed().as_secs_f64();

    let mut sink = 0i32;
    for slot in acc.iter() {
        sink = sink.wrapping_add(unsafe { vgetq_lane_s8(*slot, 0) } as i32);
    }
    ((limit as f64 * iters as f64) / secs / 1e9, sink)
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
