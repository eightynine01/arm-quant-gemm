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

/// 4×4 register-tiled `SDOT` — the fair opponent for [`gemm_smmla_8x8`].
///
/// [`gemm_sdot`] computes one output element at a time: two loads, one `SDOT`.
/// Comparing that against a tiled `SMMLA` measures tiling, not the instruction.
/// This version loads four A rows and four B columns and issues sixteen `SDOT`s
/// from them — the same 2 instructions per load as the 8×8 `SMMLA` kernel.
///
/// With both kernels tiled, the remaining difference should be close to the
/// instruction ratio: `SDOT` does 16 MACs (4 lanes × 4), `SMMLA` does 32
/// (2×2 output × 8 deep). Anything much beyond 2× would mean something else
/// is going on.
///
/// # Safety
/// Requires FEAT_DotProd. `bt` must be [`pack_b_transposed`] output with the same `kp`.
#[target_feature(enable = "dotprod")]
pub unsafe fn gemm_sdot_tiled(
    m: usize, n: usize, k: usize, kp: usize, a: &[i8], bt: &[i8], c: &mut [i32],
) {
    let kb = k / 16 * 16;

    let mut i0 = 0;
    while i0 < m {
        let mut j0 = 0;
        while j0 < n {
            let mut acc = [[vdupq_n_s32(0); 4]; 4];

            // Clamp so pointers stay in bounds; out-of-range tiles are not stored.
            let mut ap = [std::ptr::null::<i8>(); 4];
            let mut bp = [std::ptr::null::<i8>(); 4];
            for t in 0..4 {
                ap[t] = a.as_ptr().add((i0 + t).min(m - 1) * k);
                bp[t] = bt.as_ptr().add((j0 + t).min(n - 1) * kp);
            }

            let mut p = 0;
            while p < kb {
                let av = [
                    vld1q_s8(ap[0].add(p)), vld1q_s8(ap[1].add(p)),
                    vld1q_s8(ap[2].add(p)), vld1q_s8(ap[3].add(p)),
                ];
                let bv = [
                    vld1q_s8(bp[0].add(p)), vld1q_s8(bp[1].add(p)),
                    vld1q_s8(bp[2].add(p)), vld1q_s8(bp[3].add(p)),
                ];
                for r in 0..4 {
                    for cc in 0..4 {
                        acc[r][cc] = vdotq_s32(acc[r][cc], av[r], bv[cc]);
                    }
                }
                p += 16;
            }

            for r in 0..4 {
                for cc in 0..4 {
                    let (i, j) = (i0 + r, j0 + cc);
                    if i < m && j < n {
                        let mut sum = vaddvq_s32(acc[r][cc]);
                        for q in kb..k {
                            sum += *a.as_ptr().add(i * k + q) as i32
                                 * *bt.as_ptr().add(j * kp + q) as i32;
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
            j0 += 4;
        }
        i0 += 4;
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

/// Pick a kernel by shape.
///
/// This rule was built on a measurement that later shrank. With the 4×4 tile,
/// `SMMLA` lost 21% to `SDOT` at M=1 and dispatching was worth 1.27×. With the
/// 8×8 tile the two are tied at M=1 (0.93–1.02× across thread counts) and
/// dispatching is worth 1.03×.
///
/// The underlying asymmetry is real but small: `SMMLA` always emits two output
/// rows, so at M=1 the second is padding. Widening the tile makes that waste
/// worse (8×8 spans four row pairs, three of them padding at M=1) while making
/// load amortisation better, and the two roughly cancel.
///
/// Kept because it costs one integer compare and is never negative. Not kept
/// because it is important — fixing the register tile mattered ~10× more.
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

/// Above this thread count, spinning at the barrier costs more than it saves.
///
/// This machine has 16 performance cores, and the cliff sits exactly there.
/// Measured at 8x4096x4096, spin ÷ channel over three runs each:
///
/// ```text
///      8 threads   1.34  1.33  1.36
///     12           1.71  1.65  1.76
///     16           1.96  1.76  1.81
///     20           0.79  0.85  0.79   <- reverses
///     24           0.82  0.75  0.82
/// ```
///
/// Past 16 the spinners have nowhere to sit but the efficiency cores, and idle
/// spinning stops being free the moment it competes with real work.
///
/// **This constant was 12 and that was wrong.** It came from a single reading
/// where 16 threads showed spin losing 0.76x. Five later runs at the same
/// setting gave 1.28x-1.82x, all wins — near saturation the run-to-run spread
/// is wide enough that one sample decides nothing. The 20- and 24-thread
/// reversals, by contrast, repeat within 0.10x and are what the cliff actually
/// is.
pub const SPIN_MAX_THREADS: usize = 16;

/// Everything the repo measured, behind one call.
///
/// Two rules decide the work, and neither is guessable from the instruction
/// descriptions:
///
/// - **Which instruction**, by `m`. `SMMLA` does twice the MACs per instruction
///   and issues at half the rate, so peak throughput is a wash (424 vs 433
///   GOPS). Its real advantage is needing half as many instructions and loads —
///   which evaporates at `m = 1`, where half of every 2x2 tile is padding.
///   Crossover is exactly `m = 2`.
/// - **Which barrier**, by thread count. Spinning wins up to
///   [`SPIN_MAX_THREADS`] and loses past it.
///
/// The pools are owned by the engine rather than built per call, which is the
/// whole point: per-call thread creation was 43–62% of the wall clock at these
/// sizes. An `Engine` constructed inside the hot path would reintroduce exactly
/// the cost this repo spent its measurements removing.
pub struct Engine {
    spin: Option<SpinPool>,
    chan: Option<Pool>,
}

impl Engine {
    /// Build the pools once. Not free — do this outside the hot path.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        if threads == 1 {
            Engine { spin: None, chan: None }
        } else if threads <= SPIN_MAX_THREADS {
            Engine { spin: Some(SpinPool::new(threads)), chan: None }
        } else {
            Engine { spin: None, chan: Some(Pool::new(threads)) }
        }
    }

    /// # Safety
    /// Requires FEAT_I8MM and FEAT_DotProd. `pa`/`pb` must be the `SMMLA`
    /// packings and `bt` the `SDOT` packing; whichever [`choose`] selects for
    /// this `m` is the one that gets read.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(
        &self,
        m: usize, n: usize, k: usize,
        mp: usize, np: usize, kp: usize,
        a: &[i8], bt: &[i8],
        pa: &[i8], pb: &[i8],
        c: &mut [i32],
    ) {
        match choose(m) {
            Kernel::Smmla => match (&self.spin, &self.chan) {
                (Some(p), _) => gemm_smmla_spin(m, n, mp, np, kp, pa, pb, c, p),
                (None, Some(p)) => gemm_smmla_pool(m, n, mp, np, kp, pa, pb, c, p),
                _ => gemm_smmla_8x8(m, n, mp, np, kp, pa, pb, c),
            },
            Kernel::Sdot => match (&self.spin, &self.chan) {
                (Some(p), _) => gemm_sdot_spin(m, n, k, k, a, bt, c, p),
                (None, Some(p)) => gemm_sdot_pool(m, n, k, k, a, bt, c, p),
                _ => gemm_sdot_tiled(m, n, k, k, a, bt, c),
            },
        }
    }
}

/// Blocked `SMMLA`: same arithmetic, different loop order.
///
/// [`gemm_smmla`] walks every column pair for one row pair before moving on, so
/// it re-reads the whole A panel once per column pair. At 256³ the panels stay
/// resident and that costs nothing; at 1024³ they do not, and throughput drops
/// from ~202 to ~151 GOPS.
///
/// Blocking N means the A panel is loaded once per block and reused across the
/// column pairs inside it, so the re-reads hit cache instead of memory.
/// `NBLOCK` is in columns and is deliberately conservative — a packed B block of
/// `NBLOCK × kp` bytes plus the A panel should sit inside L2.
pub const NBLOCK: usize = 256;

/// # Safety
/// Requires FEAT_I8MM. Same packing contract as [`gemm_smmla`].
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_smmla_blocked(
    m: usize, n: usize, mp: usize, np: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32],
) {
    let colpairs = np / 2;
    let cp_block = (NBLOCK / 2).max(1);
    let mut cp0 = 0;
    while cp0 < colpairs {
        let cp1 = (cp0 + cp_block).min(colpairs);
        smmla_colrange(m, n, mp, kp, pa, pb, c, cp0, cp1);
        cp0 = cp1;
    }
}

/// 8×8 register-tiled `SMMLA` — the optimization the load count actually points at.
///
/// The 4×4 kernel issues four `SMMLA`s against four 16-byte loads per k-step:
/// one load per instruction. If the core sustains two 128-bit loads per cycle
/// that caps it at two `SMMLA`/cycle no matter how well anything caches, which
/// is why blocking N changed nothing.
///
/// An 8×8 tile loads four A vectors and four B vectors and issues **sixteen**
/// `SMMLA`s from them — 2 instructions per load instead of 1. Cost is register
/// pressure: 16 accumulators plus 8 operands is 24 of the 32 NEON registers,
/// which fits, but leaves little room for the compiler to spill into.
///
/// # Safety
/// Requires FEAT_I8MM. Same packing contract as [`gemm_smmla`].
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_smmla_8x8(
    m: usize, n: usize, mp: usize, np: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32],
) {
    smmla_8x8_colrange(m, n, mp, kp, pa, pb, c, 0, np / 2);
}

/// 8×8 tiling restricted to `[cp0_lo, cp1_hi)` column pairs, so the threaded
/// path gets the same register tiling as the single-threaded one.
///
/// # Safety
/// Requires FEAT_I8MM. Same packing contract as [`gemm_smmla`].
#[target_feature(enable = "i8mm")]
pub unsafe fn smmla_8x8_colrange(
    m: usize, n: usize, mp: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32], cp0_lo: usize, cp1_hi: usize,
) {
    let kblocks = kp / 8;
    let rowpairs = mp / 2;
    let colpairs = cp1_hi;

    let mut rp0 = 0;
    while rp0 < rowpairs {
        let mut cp0 = cp0_lo;
        while cp0 < cp1_hi {
            let mut acc = [[vdupq_n_s32(0); 4]; 4];

            // Clamp so the pointer arithmetic stays inside the packed panels.
            // Tiles past the real edge compute garbage; `store_tile` drops them
            // because their row/col indices are >= m/n.
            let mut ap = [std::ptr::null::<i8>(); 4];
            let mut bp = [std::ptr::null::<i8>(); 4];
            for i in 0..4 {
                ap[i] = pa.as_ptr().add((rp0 + i).min(rowpairs - 1) * kblocks * 16);
                bp[i] = pb.as_ptr().add((cp0 + i).min(colpairs - 1) * kblocks * 16);
            }

            for kb in 0..kblocks {
                let off = kb * 16;
                let av = [
                    vld1q_s8(ap[0].add(off)), vld1q_s8(ap[1].add(off)),
                    vld1q_s8(ap[2].add(off)), vld1q_s8(ap[3].add(off)),
                ];
                let bv = [
                    vld1q_s8(bp[0].add(off)), vld1q_s8(bp[1].add(off)),
                    vld1q_s8(bp[2].add(off)), vld1q_s8(bp[3].add(off)),
                ];
                for i in 0..4 {
                    for j in 0..4 {
                        acc[i][j] = vmmlaq_s32(acc[i][j], av[i], bv[j]);
                    }
                }
            }

            for i in 0..4 {
                for j in 0..4 {
                    if rp0 + i < rowpairs && cp0 + j < colpairs {
                        store_tile(acc[i][j], (rp0 + i) * 2, (cp0 + j) * 2, m, n, c);
                    }
                }
            }
            cp0 += 4;
        }
        rp0 += 4;
    }
}

/// Same tile, same traffic, `k` unrolled by two.
///
/// The roofline says this kernel is memory-bound but only reaches 26% of its
/// memory ceiling, and that B is already read the minimum number of times — so
/// the gap is not traffic volume but how much of it is in flight at once. The
/// `kb` loop issues eight independent loads and then consumes them; doubling
/// the step puts sixteen in flight against the same bytes.
///
/// Kept as a separate function rather than replacing the original so the two
/// can be measured against each other. `unroll_demo()` reports the result.
///
/// # Safety
/// Requires FEAT_I8MM. Same packing contract as [`gemm_smmla`].
#[target_feature(enable = "i8mm")]
pub unsafe fn smmla_8x8_colrange_u2(
    m: usize, n: usize, mp: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32], cp0_lo: usize, cp1_hi: usize,
) {
    let kblocks = kp / 8;
    let rowpairs = mp / 2;
    let colpairs = cp1_hi;

    let mut rp0 = 0;
    while rp0 < rowpairs {
        let mut cp0 = cp0_lo;
        while cp0 < cp1_hi {
            let mut acc = [[vdupq_n_s32(0); 4]; 4];

            let mut ap = [std::ptr::null::<i8>(); 4];
            let mut bp = [std::ptr::null::<i8>(); 4];
            for i in 0..4 {
                ap[i] = pa.as_ptr().add((rp0 + i).min(rowpairs - 1) * kblocks * 16);
                bp[i] = pb.as_ptr().add((cp0 + i).min(colpairs - 1) * kblocks * 16);
            }

            let pairs = kblocks / 2;
            for kk in 0..pairs {
                let o0 = kk * 32;
                let o1 = o0 + 16;
                // Both halves' loads are issued before either is consumed, so
                // sixteen loads are outstanding instead of eight.
                let av0 = [
                    vld1q_s8(ap[0].add(o0)), vld1q_s8(ap[1].add(o0)),
                    vld1q_s8(ap[2].add(o0)), vld1q_s8(ap[3].add(o0)),
                ];
                let bv0 = [
                    vld1q_s8(bp[0].add(o0)), vld1q_s8(bp[1].add(o0)),
                    vld1q_s8(bp[2].add(o0)), vld1q_s8(bp[3].add(o0)),
                ];
                let av1 = [
                    vld1q_s8(ap[0].add(o1)), vld1q_s8(ap[1].add(o1)),
                    vld1q_s8(ap[2].add(o1)), vld1q_s8(ap[3].add(o1)),
                ];
                let bv1 = [
                    vld1q_s8(bp[0].add(o1)), vld1q_s8(bp[1].add(o1)),
                    vld1q_s8(bp[2].add(o1)), vld1q_s8(bp[3].add(o1)),
                ];
                for i in 0..4 {
                    for j in 0..4 {
                        acc[i][j] = vmmlaq_s32(acc[i][j], av0[i], bv0[j]);
                        acc[i][j] = vmmlaq_s32(acc[i][j], av1[i], bv1[j]);
                    }
                }
            }
            // Odd tail: kblocks is not required to be even.
            if kblocks % 2 == 1 {
                let off = (kblocks - 1) * 16;
                let av = [
                    vld1q_s8(ap[0].add(off)), vld1q_s8(ap[1].add(off)),
                    vld1q_s8(ap[2].add(off)), vld1q_s8(ap[3].add(off)),
                ];
                let bv = [
                    vld1q_s8(bp[0].add(off)), vld1q_s8(bp[1].add(off)),
                    vld1q_s8(bp[2].add(off)), vld1q_s8(bp[3].add(off)),
                ];
                for i in 0..4 {
                    for j in 0..4 {
                        acc[i][j] = vmmlaq_s32(acc[i][j], av[i], bv[j]);
                    }
                }
            }

            for i in 0..4 {
                for j in 0..4 {
                    if rp0 + i < rowpairs && cp0 + j < colpairs {
                        store_tile(acc[i][j], (rp0 + i) * 2, (cp0 + j) * 2, m, n, c);
                    }
                }
            }
            cp0 += 4;
        }
        rp0 += 4;
    }
}

/// `k`-unrolled 8×8 over the full column range.
///
/// # Safety
/// Requires FEAT_I8MM. Same packing contract as [`gemm_smmla`].
#[target_feature(enable = "i8mm")]
pub unsafe fn gemm_smmla_8x8_u2(
    m: usize, n: usize, mp: usize, np: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32],
) {
    smmla_8x8_colrange_u2(m, n, mp, kp, pa, pb, c, 0, np / 2);
}

/// Worker threads that outlive a single call.
///
/// `std::thread::scope` creates fresh OS threads every time it is entered. At
/// `8x4096x4096` the whole GEMM is ~0.3ms and spawning 16 threads is ~0.16ms of
/// that, so past roughly 8-12 threads the creation cost grows faster than the
/// per-thread work shrinks and adding threads makes things slower. Measured in
/// `thread_balance_demo()`.
///
/// The workers here are created once and parked on a channel.
pub struct Pool {
    txs: Vec<std::sync::mpsc::Sender<Job>>,
    done: std::sync::mpsc::Receiver<()>,
    n: usize,
}

/// A worker index and an erased pointer to the closure to run.
struct Job(usize, *const (dyn Fn(usize) + Sync + 'static));

/// Sound because [`Pool::run`] blocks until every worker has finished with the
/// pointer, so the closure it borrows outlives the send. Nothing else can
/// observe the pointer.
unsafe impl Send for Job {}

impl Pool {
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
        let (done_tx, done) = std::sync::mpsc::channel();
        let mut txs = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            let dt = done_tx.clone();
            std::thread::spawn(move || {
                // Exits when the Pool is dropped and the sender closes.
                while let Ok(Job(idx, f)) = rx.recv() {
                    unsafe { (*f)(idx) };
                    if dt.send(()).is_err() {
                        break;
                    }
                }
            });
            txs.push(tx);
        }
        Pool { txs, done, n }
    }

    pub fn threads(&self) -> usize {
        self.n
    }

    /// Run `f(i)` on worker `i` for every worker, and block until all return.
    ///
    /// The lifetime erasure is the whole trick, and the blocking join below is
    /// what makes it sound: `f` cannot be dropped while a worker still holds
    /// the pointer, because this function does not return until every worker
    /// has sent its completion.
    ///
    /// A panicking worker would leave this waiting forever. That is acceptable
    /// here — the kernels are panic-free and a benchmark that hangs is more
    /// honest than one that silently reports a partial result.
    pub fn run<F: Fn(usize) + Sync>(&self, f: F) {
        let borrowed: &(dyn Fn(usize) + Sync) = &f;
        let erased: *const (dyn Fn(usize) + Sync + 'static) =
            unsafe { std::mem::transmute(borrowed) };
        for (i, tx) in self.txs.iter().enumerate() {
            tx.send(Job(i, erased)).expect("pool worker died");
        }
        for _ in 0..self.n {
            self.done.recv().expect("pool worker died");
        }
    }
}

/// `SMMLA` over a persistent pool. Same partition as [`gemm_smmla_mt`], so any
/// difference between the two is thread management and nothing else.
///
/// # Safety
/// Requires FEAT_I8MM. Same packing contract as [`gemm_smmla`].
pub unsafe fn gemm_smmla_pool(
    m: usize, n: usize, mp: usize, np: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32], pool: &Pool,
) {
    let colpairs = np / 2;
    let nthreads = pool.threads().min(colpairs.max(1));
    if nthreads <= 1 {
        gemm_smmla_8x8(m, n, mp, np, kp, pa, pb, c);
        return;
    }
    let cp_chunk = colpairs.div_ceil(nthreads);
    let cptr = CPtr(c.as_mut_ptr());
    let clen = c.len();

    pool.run(|t| {
        let cp0 = t * cp_chunk;
        if cp0 >= colpairs {
            return;
        }
        let cp1 = ((t + 1) * cp_chunk).min(colpairs);
        let cptr = cptr;
        unsafe {
            let cslice = std::slice::from_raw_parts_mut(cptr.0, clen);
            smmla_8x8_colrange(m, n, mp, kp, pa, pb, cslice, cp0, cp1);
        }
    });
}

/// The same idea with the channels taken out.
///
/// `Pool` fixed thread *creation*, and measuring again showed dispatch is still
/// 40.5–41.8% of a call: an mpsc send and recv per worker per job, on a job
/// that is ~0.1ms long. Shrinking the B panel from 16 MiB to 1 MiB moved
/// nothing (2379.8 vs 2307.2 GOPS), so the remainder is synchronisation and not
/// memory.
///
/// Workers here spin on an atomic generation counter instead. Spinning burns
/// cores while idle, which is the right trade only because dispatch is
/// sub-millisecond and the pool is busy whenever it exists.
pub struct SpinPool {
    shared: std::sync::Arc<SpinShared>,
    n: usize,
}

struct SpinShared {
    /// Bumped once per job. Workers wake when it changes.
    gen: std::sync::atomic::AtomicUsize,
    /// Workers that have finished the current generation.
    done: std::sync::atomic::AtomicUsize,
    stop: std::sync::atomic::AtomicBool,
    /// Written before `gen` is bumped (Release) and read after it is observed
    /// (Acquire), so the counter is what publishes it.
    job: std::cell::UnsafeCell<Option<*const (dyn Fn(usize) + Sync + 'static)>>,
}

/// Sound because `job` is only written while every worker is parked — `run`
/// does not return until `done` reaches `n` — and the generation counter
/// carries the release/acquire edge that publishes it.
unsafe impl Sync for SpinShared {}
unsafe impl Send for SpinShared {}

impl SpinPool {
    pub fn new(n: usize) -> Self {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let n = n.max(1);
        let shared = std::sync::Arc::new(SpinShared {
            gen: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: std::cell::UnsafeCell::new(None),
        });
        // Worker 0 is the calling thread, so only n-1 are spawned.
        for i in 1..n {
            let sh = shared.clone();
            std::thread::spawn(move || {
                let mut seen = 0usize;
                loop {
                    let mut spins = 0u32;
                    loop {
                        if sh.stop.load(Ordering::Acquire) {
                            return;
                        }
                        let g = sh.gen.load(Ordering::Acquire);
                        if g != seen {
                            seen = g;
                            break;
                        }
                        spins += 1;
                        // Spin briefly, then yield: a pool that outlives the
                        // benchmark should not pin cores at 100% forever.
                        if spins < 4096 {
                            std::hint::spin_loop();
                        } else {
                            std::thread::yield_now();
                        }
                    }
                    if let Some(f) = unsafe { *sh.job.get() } {
                        unsafe { (*f)(i) };
                    }
                    sh.done.fetch_add(1, Ordering::Release);
                }
            });
        }
        SpinPool { shared, n }
    }

    pub fn threads(&self) -> usize {
        self.n
    }

    /// Run `f(i)` for every worker index, with the caller acting as worker 0.
    pub fn run<F: Fn(usize) + Sync>(&self, f: F) {
        use std::sync::atomic::Ordering;
        let borrowed: &(dyn Fn(usize) + Sync) = &f;
        let erased: *const (dyn Fn(usize) + Sync + 'static) =
            unsafe { std::mem::transmute(borrowed) };

        unsafe { *self.shared.job.get() = Some(erased) };
        self.shared.done.store(0, Ordering::Relaxed);
        self.shared.gen.fetch_add(1, Ordering::Release);

        // The caller does its own share instead of idling through the barrier.
        f(0);

        let want = self.n - 1;
        let mut spins = 0u32;
        while self.shared.done.load(Ordering::Acquire) < want {
            spins += 1;
            if spins < 4096 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        unsafe { *self.shared.job.get() = None };
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared
            .stop
            .store(true, std::sync::atomic::Ordering::Release);
        self.shared
            .gen
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}

/// `SMMLA` over the spin pool. Identical partition to [`gemm_smmla_pool`].
///
/// # Safety
/// Requires FEAT_I8MM. Same packing contract as [`gemm_smmla`].
pub unsafe fn gemm_smmla_spin(
    m: usize, n: usize, mp: usize, np: usize, kp: usize,
    pa: &[i8], pb: &[i8], c: &mut [i32], pool: &SpinPool,
) {
    let colpairs = np / 2;
    let nthreads = pool.threads().min(colpairs.max(1));
    if nthreads <= 1 {
        gemm_smmla_8x8(m, n, mp, np, kp, pa, pb, c);
        return;
    }
    let cp_chunk = colpairs.div_ceil(nthreads);
    let cptr = CPtr(c.as_mut_ptr());
    let clen = c.len();

    pool.run(|t| {
        let cp0 = t * cp_chunk;
        if cp0 >= colpairs {
            return;
        }
        let cp1 = ((t + 1) * cp_chunk).min(colpairs);
        let cptr = cptr;
        unsafe {
            let cslice = std::slice::from_raw_parts_mut(cptr.0, clen);
            smmla_8x8_colrange(m, n, mp, kp, pa, pb, cslice, cp0, cp1);
        }
    });
}

/// `SDOT` over the spin pool.
///
/// Exists so an [`Engine`] never has to keep two pool types alive at once.
/// Idle spinners take cores from whatever else is running — measured: a channel
/// pool timed next to a live spin pool reads ~25% slow — so an engine holds
/// exactly one pool and both instruction paths have to be able to use it.
///
/// # Safety
/// Requires FEAT_DotProd. Same packing contract as [`gemm_sdot`].
pub unsafe fn gemm_sdot_spin(
    m: usize, n: usize, k: usize, kp: usize,
    a: &[i8], bt: &[i8], c: &mut [i32], pool: &SpinPool,
) {
    let nthreads = pool.threads().min(n.max(1));
    if nthreads <= 1 {
        gemm_sdot_tiled(m, n, k, kp, a, bt, c);
        return;
    }
    let chunk = n.div_ceil(nthreads);
    let cptr = CPtr(c.as_mut_ptr());
    let clen = c.len();

    pool.run(|t| {
        let c0 = t * chunk;
        if c0 >= n {
            return;
        }
        let c1 = ((t + 1) * chunk).min(n);
        let cptr = cptr;
        unsafe {
            let cslice = std::slice::from_raw_parts_mut(cptr.0, clen);
            sdot_colrange(m, n, k, kp, a, bt, cslice, c0, c1);
        }
    });
}

/// `SDOT` over the same persistent pool.
///
/// Needed because the dispatch claim — that `SDOT` wins at M=1 — was measured
/// with both sides paying per-call thread creation. If that cost dominated,
/// the comparison was partly measuring thread management rather than the
/// instructions, so both sides have to be re-measured on equal, pooled footing.
///
/// # Safety
/// Requires FEAT_DotProd. Same packing contract as [`gemm_sdot`].
pub unsafe fn gemm_sdot_pool(
    m: usize, n: usize, k: usize, kp: usize,
    a: &[i8], bt: &[i8], c: &mut [i32], pool: &Pool,
) {
    let nthreads = pool.threads().min(n.max(1));
    if nthreads <= 1 {
        gemm_sdot_tiled(m, n, k, kp, a, bt, c);
        return;
    }
    let chunk = n.div_ceil(nthreads);
    let cptr = CPtr(c.as_mut_ptr());
    let clen = c.len();

    pool.run(|t| {
        let c0 = t * chunk;
        if c0 >= n {
            return;
        }
        let c1 = ((t + 1) * chunk).min(n);
        let cptr = cptr;
        unsafe {
            let cslice = std::slice::from_raw_parts_mut(cptr.0, clen);
            sdot_colrange(m, n, k, kp, a, bt, cslice, c0, c1);
        }
    });
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
        gemm_smmla_8x8(m, n, mp, np, kp, pa, pb, c);
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
                smmla_8x8_colrange(m, n, mp, kp, pa, pb, cslice, cp0, cp1);
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
        gemm_sdot_tiled(m, n, k, kp, a, bt, c);
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

/// Column-range worker. Uses the same 4×4 register tiling as
/// [`gemm_sdot_tiled`] so the threaded comparison stays fair.
#[target_feature(enable = "dotprod")]
unsafe fn sdot_colrange(
    m: usize, n: usize, k: usize, kp: usize,
    a: &[i8], bt: &[i8], c: &mut [i32], j0: usize, j1: usize,
) {
    let kb = k / 16 * 16;
    let mut i0 = 0;
    while i0 < m {
        let mut jj = j0;
        while jj < j1 {
            let mut acc = [[vdupq_n_s32(0); 4]; 4];
            let mut ap = [std::ptr::null::<i8>(); 4];
            let mut bp = [std::ptr::null::<i8>(); 4];
            for t in 0..4 {
                ap[t] = a.as_ptr().add((i0 + t).min(m - 1) * k);
                bp[t] = bt.as_ptr().add((jj + t).min(j1 - 1) * kp);
            }
            let mut p = 0;
            while p < kb {
                let av = [vld1q_s8(ap[0].add(p)), vld1q_s8(ap[1].add(p)),
                          vld1q_s8(ap[2].add(p)), vld1q_s8(ap[3].add(p))];
                let bv = [vld1q_s8(bp[0].add(p)), vld1q_s8(bp[1].add(p)),
                          vld1q_s8(bp[2].add(p)), vld1q_s8(bp[3].add(p))];
                for r in 0..4 {
                    for cc in 0..4 {
                        acc[r][cc] = vdotq_s32(acc[r][cc], av[r], bv[cc]);
                    }
                }
                p += 16;
            }
            for r in 0..4 {
                for cc in 0..4 {
                    let (i, j) = (i0 + r, jj + cc);
                    if i < m && j < j1 {
                        let mut sum = vaddvq_s32(acc[r][cc]);
                        for q in kb..k {
                            sum += *a.as_ptr().add(i * k + q) as i32
                                 * *bt.as_ptr().add(j * kp + q) as i32;
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
            jj += 4;
        }
        i0 += 4;
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
