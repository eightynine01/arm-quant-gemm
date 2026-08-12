//! Both halves of the roofline: the instruction-issue ceiling and the memory
//! ceiling.
//!
//! The benchmark reports absolute GOPS, which says nothing about whether that
//! is good. The issue ceiling is a loop of back-to-back `SMMLA`s over
//! register-resident operands, with enough independent accumulators to hide
//! latency. Whatever that reaches is the issue limit.
//!
//! One ceiling is not enough to place a kernel: a number far below the issue
//! limit can mean wasted issue slots or it can mean the kernel is at its
//! *memory* limit, where no amount of instruction scheduling would move it.
//! Which one applies is decided by arithmetic intensity.
//!
//! For an `m x n x k` product with an `mt x nt` register tile, A is re-read once
//! per column block and B once per row block, so `2*m*n*k` ops ride on
//! `m*n*k*(mt+nt)/(mt*nt)` bytes:
//!
//! ```text
//!     intensity = 2*mt*nt/(mt+nt)   ops per byte
//! ```
//!
//! which is 8 for the 8x8 tile. At `m = 8` there is only one row block, so A is
//! read once (32 KiB, negligible) and the figure rises to `2*mt` = 16.
//! `stream_bandwidth` sweeps footprints rather than reporting one number,
//! because the bandwidth that matters is whichever level of the hierarchy the B
//! panel occupies.
//!
//! **On this machine the memory ceiling is not the binding one.** It lands near
//! 1240 GOPS against an issue ceiling of ~424, and the kernel sits at 85% of the
//! issue ceiling — so this GEMM is issue-bound, not memory-bound. An earlier
//! version of this file concluded the opposite, from an issue ceiling that was
//! being measured far too high; the k-unroll built to exploit that conclusion
//! moved nothing, which is what prompted the remeasurement.

use std::arch::aarch64::*;
use std::hint::black_box;
use std::time::Instant;

/// Written in inline assembly, after four attempts in intrinsics all failed.
///
/// The failures are worth naming because each produced a plausible number
/// rather than an error:
///
/// | attempt | result | what actually happened |
/// |---|---|---|
/// | constant operands | `inf` | loop folded away entirely |
/// | `black_box` on accumulators | 173% of peak | dependency forced through memory; measured latency |
/// | 16 identical chains | 15.7 SMMLA/cycle | all chains computed the same value, CSE'd into one |
/// | `black_box` on operand arrays | 74 GOPS vs a 380 GOPS kernel | an 8-vector array does not fit in registers, so the barrier spilled it to stack every iteration |
///
/// The last one is the reason for dropping to assembly. `black_box` is the only
/// portable barrier available, and applying it to anything larger than a single
/// register forces a round trip through the stack — which is precisely the
/// traffic a register-resident benchmark must not have. There is no placement
/// of it that both keeps eight chains independent and keeps them in registers.
///
/// In assembly the question does not arise: eight independent `SMMLA`s on
/// distinct destination registers, a `subs`/`b.ne` loop, and nothing the
/// optimiser is permitted to touch.
///
/// # Safety
/// Requires FEAT_I8MM.
#[target_feature(enable = "i8mm")]
pub unsafe fn smmla_issue_ceiling(iters: u64) -> (f64, i32) {
    let n = iters;
    let t0 = Instant::now();
    core::arch::asm!(
        // 16 accumulators (v0-v15) and 8 operands (v16-v23), initialised
        // in-place so nothing has to be handed in and no load appears in the
        // loop body.
        "movi v0.4s, #0", "movi v1.4s, #0", "movi v2.4s, #0", "movi v3.4s, #0",
        "movi v4.4s, #0", "movi v5.4s, #0", "movi v6.4s, #0", "movi v7.4s, #0",
        "movi v8.4s, #0", "movi v9.4s, #0", "movi v10.4s, #0", "movi v11.4s, #0",
        "movi v12.4s, #0", "movi v13.4s, #0", "movi v14.4s, #0", "movi v15.4s, #0",
        "movi v16.16b, #1", "movi v17.16b, #2", "movi v18.16b, #3",
        "movi v19.16b, #4", "movi v20.16b, #5", "movi v21.16b, #6",
        "movi v22.16b, #7", "movi v23.16b, #8",
        "2:",
        // 32 SMMLAs per iteration against 2 loop-control instructions. The
        // 8-per-iteration version spent ~20% of its slots on `subs`/`b.ne` and
        // came out at 262 GOPS, below the 363 GOPS kernel it was supposed to
        // bound. Each accumulator is written twice, spaced 16 instructions
        // apart, so the second write is far past the first one's latency.
        "smmla v0.4s, v16.16b, v17.16b", "smmla v1.4s, v18.16b, v19.16b",
        "smmla v2.4s, v20.16b, v21.16b", "smmla v3.4s, v22.16b, v23.16b",
        "smmla v4.4s, v16.16b, v19.16b", "smmla v5.4s, v18.16b, v21.16b",
        "smmla v6.4s, v20.16b, v23.16b", "smmla v7.4s, v22.16b, v17.16b",
        "smmla v8.4s, v16.16b, v21.16b", "smmla v9.4s, v18.16b, v23.16b",
        "smmla v10.4s, v20.16b, v17.16b", "smmla v11.4s, v22.16b, v19.16b",
        "smmla v12.4s, v16.16b, v23.16b", "smmla v13.4s, v18.16b, v17.16b",
        "smmla v14.4s, v20.16b, v19.16b", "smmla v15.4s, v22.16b, v21.16b",
        "smmla v0.4s, v17.16b, v16.16b", "smmla v1.4s, v19.16b, v18.16b",
        "smmla v2.4s, v21.16b, v20.16b", "smmla v3.4s, v23.16b, v22.16b",
        "smmla v4.4s, v19.16b, v16.16b", "smmla v5.4s, v21.16b, v18.16b",
        "smmla v6.4s, v23.16b, v20.16b", "smmla v7.4s, v17.16b, v22.16b",
        "smmla v8.4s, v21.16b, v16.16b", "smmla v9.4s, v23.16b, v18.16b",
        "smmla v10.4s, v17.16b, v20.16b", "smmla v11.4s, v19.16b, v22.16b",
        "smmla v12.4s, v23.16b, v16.16b", "smmla v13.4s, v17.16b, v18.16b",
        "smmla v14.4s, v19.16b, v20.16b", "smmla v15.4s, v21.16b, v22.16b",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        // The counter is consumed by the loop; discard the output.
        n = inout(reg) n => _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        options(nostack)
    );
    let secs = t0.elapsed().as_secs_f64();

    // Each SMMLA produces a 2x2 int32 tile from 8-deep operands: 32 MACs,
    // counted as 64 ops to match the 2*M*N*K convention used elsewhere.
    let ops = iters as f64 * 32.0 * 64.0;
    (ops / secs / 1e9, 0)
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
    let n = iters;
    let t0 = Instant::now();
    core::arch::asm!(
        "movi v0.4s, #0", "movi v1.4s, #0", "movi v2.4s, #0", "movi v3.4s, #0",
        "movi v4.4s, #0", "movi v5.4s, #0", "movi v6.4s, #0", "movi v7.4s, #0",
        "movi v8.4s, #0", "movi v9.4s, #0", "movi v10.4s, #0", "movi v11.4s, #0",
        "movi v12.4s, #0", "movi v13.4s, #0", "movi v14.4s, #0", "movi v15.4s, #0",
        "movi v16.16b, #1", "movi v17.16b, #2", "movi v18.16b, #3",
        "movi v19.16b, #4", "movi v20.16b, #5", "movi v21.16b, #6",
        "movi v22.16b, #7", "movi v23.16b, #8",
        "2:",
        "sdot v0.4s, v16.16b, v17.16b", "sdot v1.4s, v18.16b, v19.16b",
        "sdot v2.4s, v20.16b, v21.16b", "sdot v3.4s, v22.16b, v23.16b",
        "sdot v4.4s, v16.16b, v19.16b", "sdot v5.4s, v18.16b, v21.16b",
        "sdot v6.4s, v20.16b, v23.16b", "sdot v7.4s, v22.16b, v17.16b",
        "sdot v8.4s, v16.16b, v21.16b", "sdot v9.4s, v18.16b, v23.16b",
        "sdot v10.4s, v20.16b, v17.16b", "sdot v11.4s, v22.16b, v19.16b",
        "sdot v12.4s, v16.16b, v23.16b", "sdot v13.4s, v18.16b, v17.16b",
        "sdot v14.4s, v20.16b, v19.16b", "sdot v15.4s, v22.16b, v21.16b",
        "sdot v0.4s, v17.16b, v16.16b", "sdot v1.4s, v19.16b, v18.16b",
        "sdot v2.4s, v21.16b, v20.16b", "sdot v3.4s, v23.16b, v22.16b",
        "sdot v4.4s, v19.16b, v16.16b", "sdot v5.4s, v21.16b, v18.16b",
        "sdot v6.4s, v23.16b, v20.16b", "sdot v7.4s, v17.16b, v22.16b",
        "sdot v8.4s, v21.16b, v16.16b", "sdot v9.4s, v23.16b, v18.16b",
        "sdot v10.4s, v17.16b, v20.16b", "sdot v11.4s, v19.16b, v22.16b",
        "sdot v12.4s, v23.16b, v16.16b", "sdot v13.4s, v17.16b, v18.16b",
        "sdot v14.4s, v19.16b, v20.16b", "sdot v15.4s, v21.16b, v22.16b",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        // The counter is consumed by the loop; discard the output.
        n = inout(reg) n => _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        options(nostack)
    );
    let secs = t0.elapsed().as_secs_f64();

    // SDOT: 4 lanes x 4 products = 16 MACs = 32 ops.
    let ops = iters as f64 * 32.0 * 32.0;
    (ops / secs / 1e9, 0)
}
