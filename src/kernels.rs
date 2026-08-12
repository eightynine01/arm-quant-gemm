//! int8 × int8 → int32 GEMM kernels for AArch64.
//!
//! Three implementations of the same computation, so the speedup is measured
//! against an identical result rather than a different problem:
//!
//! 1. `gemm_scalar` — portable triple loop. The reference.
//! 2. `gemm_sdot`   — NEON `SDOT` (Armv8.2 FEAT_DotProd). 4 MACs per i32 lane.
//! 3. `gemm_smmla`  — `SMMLA` (Armv8.6 FEAT_I8MM). 8 MACs per i32 lane.
//!
//! `SMMLA` computes a 2×2 int32 tile from a 2×8 int8 A-tile and an 8×2 int8
//! B-tile held in one register each. That is twice the arithmetic per
//! instruction of `SDOT`, but only if the operands are already packed into
//! that layout — so the packing cost is what decides the real gain.

use std::arch::aarch64::*;

/// Row-major `C[m][n] = sum_k A[m][k] * B[k][n]`, the reference result.
pub fn gemm_scalar(m: usize, n: usize, k: usize, a: &[i8], b: &[i8], c: &mut [i32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += a[i * k + p] as i32 * b[p * n + j] as i32;
            }
            c[i * n + j] = acc;
        }
    }
}

/// A packed as row-major panels of 8 contiguous k values per row.
/// B packed transposed so each column's k values are contiguous — SDOT needs
/// both operands' k axis contiguous.
pub fn pack_b_transposed(n: usize, k: usize, kp: usize, b: &[i8]) -> Vec<i8> {
    let mut out = vec![0i8; n * kp];
    for j in 0..n {
        for p in 0..k {
            out[j * kp + p] = b[p * n + j];
        }
    }
    out
}

/// NEON `SDOT`: 16 int8 lanes → 4 int32 lanes, 4 products each.
///
/// # Safety
/// Requires FEAT_DotProd. `bt` must be `pack_b_transposed` output with the same `kp`.
#[target_feature(enable = "dotprod")]
pub unsafe fn gemm_sdot(m: usize, n: usize, k: usize, kp: usize, a: &[i8], bt: &[i8], c: &mut [i32]) {
    let kb = k / 16 * 16;
    for i in 0..m {
        for j in 0..n {
            let arow = a.as_ptr().add(i * k);
            let brow = bt.as_ptr().add(j * kp);
            let mut acc = vdupq_n_s32(0);
            let mut p = 0;
            while p < kb {
                acc = vdotq_s32(acc, vld1q_s8(arow.add(p)), vld1q_s8(brow.add(p)));
                p += 16;
            }
            let mut tail = vaddvq_s32(acc);
            for q in kb..k {
                tail += *arow.add(q) as i32 * *brow.add(q) as i32;
            }
            c[i * n + j] = tail;
        }
    }
}

/// Pack A into 2-row × 8-deep tiles: `[row0 k0..k7, row1 k0..k7]` per 16 bytes.
/// This is exactly the operand layout `SMMLA` expects.
pub fn pack_a_smmla(m: usize, k: usize, a: &[i8]) -> (Vec<i8>, usize, usize) {
    let mp = m.div_ceil(2) * 2;
    let kp = k.div_ceil(8) * 8;
    let mut out = vec![0i8; mp * kp];
    for rp in 0..mp / 2 {
        for kbi in 0..kp / 8 {
            let base = (rp * (kp / 8) + kbi) * 16;
            for r in 0..2 {
                let row = rp * 2 + r;
                for e in 0..8 {
                    let kk = kbi * 8 + e;
                    if row < m && kk < k {
                        out[base + r * 8 + e] = a[row * k + kk];
                    }
                }
            }
        }
    }
    (out, mp, kp)
}

/// Pack B into 2-col × 8-deep tiles: `[col0 k0..k7, col1 k0..k7]` per 16 bytes.
pub fn pack_b_smmla(n: usize, k: usize, b: &[i8]) -> (Vec<i8>, usize) {
    let np = n.div_ceil(2) * 2;
    let kp = k.div_ceil(8) * 8;
    let mut out = vec![0i8; np * kp];
    for cp in 0..np / 2 {
        for kbi in 0..kp / 8 {
            let base = (cp * (kp / 8) + kbi) * 16;
            for cc in 0..2 {
                let col = cp * 2 + cc;
                for e in 0..8 {
                    let kk = kbi * 8 + e;
                    if col < n && kk < k {
                        out[base + cc * 8 + e] = b[kk * n + col];
                    }
                }
            }
        }
    }
    (out, np)
}

/// `SMMLA` kernel over a 4×4 output tile — four independent accumulators so the
/// four SMMLAs in each k-step can issue back to back instead of serialising on
/// one accumulator.
///
/// # Safety
/// Requires FEAT_I8MM. `pa`/`pb` must come from `pack_a_smmla`/`pack_b_smmla`.
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_smmla(
    m: usize, n: usize, mp: usize, np: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32],
) {
    let kblocks = kp / 8;
    let rowpairs = mp / 2;
    let colpairs = np / 2;

    let mut rp = 0;
    while rp < rowpairs {
        let two_rp = (rp + 1) < rowpairs;
        let mut cp = 0;
        while cp < colpairs {
            let two_cp = (cp + 1) < colpairs;

            let mut acc00 = vdupq_n_s32(0);
            let mut acc01 = vdupq_n_s32(0);
            let mut acc10 = vdupq_n_s32(0);
            let mut acc11 = vdupq_n_s32(0);

            let a0 = pa.as_ptr().add(rp * kblocks * 16);
            let a1 = pa.as_ptr().add((rp + 1).min(rowpairs - 1) * kblocks * 16);
            let b0 = pb.as_ptr().add(cp * kblocks * 16);
            let b1 = pb.as_ptr().add((cp + 1).min(colpairs - 1) * kblocks * 16);

            for kb in 0..kblocks {
                let off = kb * 16;
                let av0 = vld1q_s8(a0.add(off));
                let bv0 = vld1q_s8(b0.add(off));
                acc00 = vmmlaq_s32(acc00, av0, bv0);
                if two_cp {
                    acc01 = vmmlaq_s32(acc01, av0, vld1q_s8(b1.add(off)));
                }
                if two_rp {
                    let av1 = vld1q_s8(a1.add(off));
                    acc10 = vmmlaq_s32(acc10, av1, bv0);
                    if two_cp {
                        acc11 = vmmlaq_s32(acc11, av1, vld1q_s8(b1.add(off)));
                    }
                }
            }

            store_tile(acc00, rp * 2, cp * 2, m, n, c);
            if two_cp { store_tile(acc01, rp * 2, cp * 2 + 2, m, n, c); }
            if two_rp { store_tile(acc10, rp * 2 + 2, cp * 2, m, n, c); }
            if two_rp && two_cp { store_tile(acc11, rp * 2 + 2, cp * 2 + 2, m, n, c); }

            cp += 2;
        }
        rp += 2;
    }
}

/// Which kernel wins depends on M, and the crossover is not where you would
/// guess. `SMMLA` always produces two output rows; at M=1 the second row is
/// padding, so half the multiply-accumulate work is discarded and `SDOT` —
/// which computes exactly one row — comes out ahead despite doing half the
/// MACs per instruction.
///
/// That matters because M=1 is the LLM decode step: every token after the
/// prompt is a matrix-vector product. A kernel chosen for prefill throughput
/// is the wrong kernel for the part that runs thousands of times.
///
/// The size of the effect is thread-dependent, and measuring that mattered:
/// at M=1 the SMMLA/SDOT ratio runs 0.79× on one thread, 0.94× on eight, and
/// 1.05× on sixteen. Past roughly eight threads the decode shape is bound by
/// memory bandwidth rather than issue rate, so SMMLA's discarded half stops
/// costing anything.
///
/// The rule below stays simple anyway, because the asymmetry favours it:
/// picking SDOT at M=1 gives up at most ~5% (16 threads) and gains up to 27%
/// (1 thread). A thread-count-aware rule would recover that 5% and is not
/// worth the extra state.
pub const SMMLA_MIN_ROWS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Sdot,
    Smmla,
}

/// Pick by shape. See [`SMMLA_MIN_ROWS`].
pub fn choose(m: usize) -> Kernel {
    if m >= SMMLA_MIN_ROWS { Kernel::Smmla } else { Kernel::Sdot }
}

/// Raw `*mut i32` that threads may hold concurrently.
///
/// Sound only because every thread writes a disjoint set of C elements — the
/// column-range partition below never overlaps. Nothing reads C during the
/// parallel region.
#[derive(Clone, Copy)]
struct CPtr(*mut i32);
unsafe impl Send for CPtr {}
unsafe impl Sync for CPtr {}

/// Multi-threaded `SMMLA`, partitioned over output **columns**.
///
/// Columns rather than rows because the shape that matters most — decode, M=1 —
/// has exactly one row to give out. Partitioning rows would leave every thread
/// but one idle at precisely the size where latency is most visible.
///
/// # Safety
/// Requires FEAT_I8MM. Same packing contract as [`gemm_smmla`].
pub unsafe fn gemm_smmla_mt(
    m: usize, n: usize, mp: usize, np: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32], threads: usize,
) {
    let colpairs = np / 2;
    let nthreads = threads.max(1).min(colpairs.max(1));
    if nthreads == 1 {
        gemm_smmla(m, n, mp, np, kp, pa, pb, c);
        return;
    }
    let cp_chunk = colpairs.div_ceil(nthreads);
    let cptr = CPtr(c.as_mut_ptr());
    let clen = c.len();

    std::thread::scope(|s| {
        for t in 0..nthreads {
            let cp0 = t * cp_chunk;
            if cp0 >= colpairs {
                break;
            }
            let cp1 = ((t + 1) * cp_chunk).min(colpairs);
            s.spawn(move || {
                // Bind the whole wrapper: Rust 2021 disjoint capture would
                // otherwise capture the bare `*mut i32`, which is not Send.
                let cptr = cptr;
                let cslice = std::slice::from_raw_parts_mut(cptr.0, clen);
                smmla_colrange(m, n, mp, kp, pa, pb, cslice, cp0, cp1);
            });
        }
    });
}

/// The column-range worker. Same 4×4 tiling as [`gemm_smmla`], restricted to
/// `[cp0, cp1)` column pairs.
#[target_feature(enable = "i8mm")]
unsafe fn smmla_colrange(
    m: usize, n: usize, mp: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32], cp0: usize, cp1: usize,
) {
    let kblocks = kp / 8;
    let rowpairs = mp / 2;

    let mut rp = 0;
    while rp < rowpairs {
        let two_rp = (rp + 1) < rowpairs;
        let mut cp = cp0;
        while cp < cp1 {
            let two_cp = (cp + 1) < cp1;

            let mut acc00 = vdupq_n_s32(0);
            let mut acc01 = vdupq_n_s32(0);
            let mut acc10 = vdupq_n_s32(0);
            let mut acc11 = vdupq_n_s32(0);

            let a0 = pa.as_ptr().add(rp * kblocks * 16);
            let a1 = pa.as_ptr().add((rp + 1).min(rowpairs - 1) * kblocks * 16);
            let b0 = pb.as_ptr().add(cp * kblocks * 16);
            let b1 = pb.as_ptr().add((cp + 1).min(cp1 - 1) * kblocks * 16);

            for kb in 0..kblocks {
                let off = kb * 16;
                let av0 = vld1q_s8(a0.add(off));
                let bv0 = vld1q_s8(b0.add(off));
                acc00 = vmmlaq_s32(acc00, av0, bv0);
                if two_cp {
                    acc01 = vmmlaq_s32(acc01, av0, vld1q_s8(b1.add(off)));
                }
                if two_rp {
                    let av1 = vld1q_s8(a1.add(off));
                    acc10 = vmmlaq_s32(acc10, av1, bv0);
                    if two_cp {
                        acc11 = vmmlaq_s32(acc11, av1, vld1q_s8(b1.add(off)));
                    }
                }
            }

            store_tile(acc00, rp * 2, cp * 2, m, n, c);
            if two_cp { store_tile(acc01, rp * 2, cp * 2 + 2, m, n, c); }
            if two_rp { store_tile(acc10, rp * 2 + 2, cp * 2, m, n, c); }
            if two_rp && two_cp { store_tile(acc11, rp * 2 + 2, cp * 2 + 2, m, n, c); }

            cp += 2;
        }
        rp += 2;
    }
}

/// Multi-threaded `SDOT`, also partitioned over columns so the comparison
/// against [`gemm_smmla_mt`] is like for like.
///
/// # Safety
/// Requires FEAT_DotProd. `bt` must be [`pack_b_transposed`] output.
pub unsafe fn gemm_sdot_mt(
    m: usize, n: usize, k: usize, kp: usize,
    a: &[i8], bt: &[i8], c: &mut [i32], threads: usize,
) {
    let nthreads = threads.max(1).min(n.max(1));
    if nthreads == 1 {
        gemm_sdot(m, n, k, kp, a, bt, c);
        return;
    }
    let chunk = n.div_ceil(nthreads);
    let cptr = CPtr(c.as_mut_ptr());
    let clen = c.len();

    std::thread::scope(|s| {
        for t in 0..nthreads {
            let j0 = t * chunk;
            if j0 >= n {
                break;
            }
            let j1 = ((t + 1) * chunk).min(n);
            s.spawn(move || {
                let cptr = cptr;
                let cslice = std::slice::from_raw_parts_mut(cptr.0, clen);
                sdot_colrange(m, n, k, kp, a, bt, cslice, j0, j1);
            });
        }
    });
}

#[target_feature(enable = "dotprod")]
unsafe fn sdot_colrange(
    m: usize, n: usize, k: usize, kp: usize,
    a: &[i8], bt: &[i8], c: &mut [i32], j0: usize, j1: usize,
) {
    let kb = k / 16 * 16;
    for i in 0..m {
        for j in j0..j1 {
            let arow = a.as_ptr().add(i * k);
            let brow = bt.as_ptr().add(j * kp);
            let mut acc = vdupq_n_s32(0);
            let mut p = 0;
            while p < kb {
                acc = vdotq_s32(acc, vld1q_s8(arow.add(p)), vld1q_s8(brow.add(p)));
                p += 16;
            }
            let mut tail = vaddvq_s32(acc);
            for q in kb..k {
                tail += *arow.add(q) as i32 * *brow.add(q) as i32;
            }
            c[i * n + j] = tail;
        }
    }
}

/// SMMLA lane order is `[r0c0, r0c1, r1c0, r1c1]`. Bounds-checked because the
/// packed panels are padded to even m/n but C is not.
#[inline]
unsafe fn store_tile(acc: int32x4_t, r: usize, c0: usize, m: usize, n: usize, c: &mut [i32]) {
    let v = [
        vgetq_lane_s32(acc, 0), vgetq_lane_s32(acc, 1),
        vgetq_lane_s32(acc, 2), vgetq_lane_s32(acc, 3),
    ];
    for (idx, &val) in v.iter().enumerate() {
        let (dr, dc) = (r + idx / 2, c0 + idx % 2);
        if dr < m && dc < n {
            c[dr * n + dc] = val;
        }
    }
}
