#![feature(stdarch_neon_i8mm)]
//! Benchmark harness. Correctness is checked before any timing is reported —
//! a speedup over a wrong answer is not a speedup.

mod kernels;
mod roofline;
use kernels::*;
use std::time::Instant;

/// Deterministic pseudo-random int8 fill (xorshift) so runs are reproducible
/// and results are comparable across machines without shipping a data file.
fn fill(buf: &mut [i8], seed: u64) {
    let mut s = seed | 1;
    for v in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = (s >> 24) as i8;
    }
}

fn run_all(m: usize, n: usize, k: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
    fill(&mut a, 0x9E3779B97F4A7C15);
    fill(&mut b, 0xBF58476D1CE4E5B9);

    let mut c_ref = vec![0i32; m * n];
    gemm_scalar(m, n, k, &a, &b, &mut c_ref);

    let kp_dot = k;
    let bt = pack_b_transposed(n, k, kp_dot, &b);
    let mut c_dot = vec![0i32; m * n];
    unsafe { gemm_sdot(m, n, k, kp_dot, &a, &bt, &mut c_dot) };

    let (pa, mp, kp) = pack_a_smmla(m, k, &a);
    let (pb, np) = pack_b_smmla(n, k, &b);
    let mut c_mm = vec![0i32; m * n];
    unsafe { gemm_smmla(m, n, mp, np, kp, &pa, &pb, &mut c_mm) };

    // The blocked variant only reorders loops, so it must agree exactly.
    let mut c_bl = vec![0i32; m * n];
    unsafe { gemm_smmla_blocked(m, n, mp, np, kp, &pa, &pb, &mut c_bl) };
    assert_eq!(c_bl, c_mm, "blocked SMMLA diverged at {}x{}x{}", m, n, k);

    // Tiled SDOT must agree too - it is the fair opponent, so it must be correct.
    let mut c_dt = vec![0i32; m * n];
    unsafe { gemm_sdot_tiled(m, n, k, kp_dot, &a, &bt, &mut c_dt) };
    assert_eq!(c_dt, c_ref, "tiled SDOT diverged at {}x{}x{}", m, n, k);

    // 8x8 tiling changes register allocation and edge handling, not arithmetic.
    let mut c88 = vec![0i32; m * n];
    unsafe { gemm_smmla_8x8(m, n, mp, np, kp, &pa, &pb, &mut c88) };
    assert_eq!(c88, c_mm, "8x8 SMMLA diverged at {}x{}x{}", m, n, k);

    (c_ref, c_dot, c_mm)
}

/// Same computation through the threaded kernels. Threading is where a
/// column-partitioned kernel silently corrupts overlapping tiles, so both
/// multi-threaded paths are checked against the scalar reference too.
fn run_mt(m: usize, n: usize, k: usize, threads: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
    fill(&mut a, 0x9E3779B97F4A7C15);
    fill(&mut b, 0xBF58476D1CE4E5B9);

    let mut c_ref = vec![0i32; m * n];
    gemm_scalar(m, n, k, &a, &b, &mut c_ref);

    let bt = pack_b_transposed(n, k, k, &b);
    let mut c_dot = vec![0i32; m * n];
    unsafe { gemm_sdot_mt(m, n, k, k, &a, &bt, &mut c_dot, threads) };

    let (pa, mp, kp) = pack_a_smmla(m, k, &a);
    let (pb, np) = pack_b_smmla(n, k, &b);
    let mut c_mm = vec![0i32; m * n];
    unsafe { gemm_smmla_mt(m, n, mp, np, kp, &pa, &pb, &mut c_mm, threads) };

    (c_ref, c_dot, c_mm)
}

fn verify() -> bool {
    // Odd sizes on purpose: they exercise the padding and tail paths, which is
    // where packed kernels usually break.
    let shapes = [
        (1, 1, 8), (2, 2, 8), (3, 5, 11), (7, 7, 7), (1, 64, 128),
        (17, 33, 65), (64, 64, 64), (63, 65, 129), (128, 128, 256),
    ];
    let mut ok = true;
    println!("Correctness vs scalar reference");
    println!("┌────────────────────┬──────────┬──────────┐");
    println!("│ M×N×K              │ SDOT     │ SMMLA    │");
    println!("├────────────────────┼──────────┼──────────┤");
    for (m, n, k) in shapes {
        let (r, d, s) = run_all(m, n, k);
        let (dok, sok) = (d == r, s == r);
        ok &= dok && sok;
        println!(
            "│ {:<18} │ {:<8} │ {:<8} │",
            format!("{}×{}×{}", m, n, k),
            if dok { "match" } else { "MISMATCH" },
            if sok { "match" } else { "MISMATCH" },
        );
    }
    println!("└────────────────────┴──────────┴──────────┘");

    // Column partitioning is the likeliest source of a threading bug, so check
    // thread counts that do and do not divide the column count evenly.
    print!("Multi-threaded (vs scalar), 2/3/8/16 threads: ");
    for t in [2usize, 3, 8, 16] {
        for (m, n, k) in [(1usize, 257usize, 128usize), (5, 64, 96), (64, 129, 130)] {
            let (r, d, s) = run_mt(m, n, k, t);
            if d != r || s != r {
                println!("MISMATCH at {}t {}×{}×{}", t, m, n, k);
                return false;
            }
        }
    }
    println!("all match");
    ok
}

fn bench_one<F: FnMut()>(mut f: F, ops: f64, budget_s: f64) -> f64 {
    f(); // warm up caches and branch predictors
    let mut iters = 0u64;
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < budget_s {
        f();
        iters += 1;
    }
    let secs = t0.elapsed().as_secs_f64();
    ops * iters as f64 / secs / 1e9
}

fn main() {
    println!("arm-quant-gemm — int8 GEMM on AArch64\n");

    if !verify() {
        eprintln!("\nVerification failed - refusing to report timings.");
        std::process::exit(1);
    }
    println!("All match. Benchmarking.\n");

    println!("Throughput (GOPS = 2*M*N*K / s, higher is better)");
    println!("┌──────────────────┬──────────┬──────────┬──────────┬──────────┬─────────┐");
    println!("│ M×N×K            │ SDOT 1x1 │ SDOT 4x4 │ MM 4x4   │  MM 8x8  │ MM/SDOT │");
    println!("├──────────────────┼──────────┼──────────┼──────────┼──────────┼─────────┤");

    for &(m, n, k) in &[
        (256usize, 256usize, 256usize),
        (512, 512, 512),
        (1024, 1024, 1024),
        (1, 4096, 4096),   // decode step: batch-1 matrix-vector
        (8, 4096, 4096),   // small batch
    ] {
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
        fill(&mut a, 0x9E3779B97F4A7C15);
        fill(&mut b, 0xBF58476D1CE4E5B9);
        let mut c = vec![0i32; m * n];

        // scalar is very slow at large shapes; shrink its budget
        let sbudget = if m * n * k > 100_000_000 { 0.4 } else { 1.0 };
        let g_scalar = bench_one(|| gemm_scalar(m, n, k, &a, &b, &mut c), ops, sbudget);

        let bt = pack_b_transposed(n, k, k, &b);
        let g_dot = bench_one(|| unsafe { gemm_sdot(m, n, k, k, &a, &bt, &mut c) }, ops, 1.0);

        let (pa, mp, kp) = pack_a_smmla(m, k, &a);
        let (pb, np) = pack_b_smmla(n, k, &b);
        let g_mm = bench_one(
            || unsafe { gemm_smmla(m, n, mp, np, kp, &pa, &pb, &mut c) }, ops, 1.0);

        let g88 = bench_one(
            || unsafe { gemm_smmla_8x8(m, n, mp, np, kp, &pa, &pb, &mut c) }, ops, 1.0);
        let g_dt = bench_one(
            || unsafe { gemm_sdot_tiled(m, n, k, k, &a, &bt, &mut c) }, ops, 1.0);

        let _ = g_scalar;
        println!(
            "│ {:<16} │ {:>8.2} │ {:>8.2} │ {:>8.2} │ {:>8.2} │ {:>6.2}× │",
            format!("{}×{}×{}", m, n, k),
            g_dot, g_dt, g_mm, g88, g88 / g_dt
        );
    }
    println!("└──────────────────┴──────────┴──────────┴──────────┴──────────┴─────────┘");
    println!("Note 1: packing is outside the timed loop - matching inference, where");
    println!("        weights are packed once and reused across many tokens.");
    println!("Note 2: scalar is a naive triple loop with cache-hostile access to B. The");
    println!("        meaningful comparison is SMMLA vs SDOT; read the scalar multiples as an upper bound.");

    dispatch_demo();
    roofline_demo();
    unroll_demo();
}

/// Does putting more loads in flight close any of the gap to the memory ceiling?
///
/// The roofline localised the headroom to memory-level parallelism: traffic is
/// already minimal and issue slots are not the limit. This is the direct test —
/// same tile, same bytes, `k` unrolled by two so sixteen loads are outstanding
/// instead of eight.
fn unroll_demo() {
    println!("\nk-unroll by 2 (same tile, same traffic, 2x loads in flight)");
    println!("┌──────────────────┬──────────┬──────────┬──────────┬─────────┐");
    println!("│ M×N×K            │  8×8     │  8×8 u2  │   gain   │ correct │");
    println!("├──────────────────┼──────────┼──────────┼──────────┼─────────┤");

    for &(m, n, k) in &[
        (8usize, 4096usize, 4096usize),
        (64, 1024, 1024),
        (1024, 1024, 1024),
    ] {
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
        fill(&mut a, 0x9E3779B97F4A7C15);
        fill(&mut b, 0xBF58476D1CE4E5B9);
        let (pa, mp, kp) = pack_a_smmla(m, k, &a);
        let (pb, np) = pack_b_smmla(n, k, &b);

        // Correctness before speed: a faster kernel that computes the wrong
        // thing is not a result. Compared against the existing 8x8, which the
        // scalar reference already validates above.
        let mut c0 = vec![0i32; m * n];
        let mut c1 = vec![0i32; m * n];
        unsafe { gemm_smmla_8x8(m, n, mp, np, kp, &pa, &pb, &mut c0) };
        unsafe { gemm_smmla_8x8_u2(m, n, mp, np, kp, &pa, &pb, &mut c1) };
        let ok = c0 == c1;

        let g0 = bench_one(
            || unsafe { gemm_smmla_8x8(m, n, mp, np, kp, &pa, &pb, &mut c0) }, ops, 0.8);
        let g1 = bench_one(
            || unsafe { gemm_smmla_8x8_u2(m, n, mp, np, kp, &pa, &pb, &mut c1) }, ops, 0.8);

        println!(
            "│ {:<16} │ {:>8.1} │ {:>8.1} │ {:>7.2}× │ {:>7} │",
            format!("{}×{}×{}", m, n, k), g0, g1, g1 / g0,
            if ok { "match" } else { "MISMATCH" }
        );
    }
    println!("└──────────────────┴──────────┴──────────┴──────────┴─────────┘");
}

/// What fraction of the machine are we actually using?
///
/// Two ceilings, not one. Reporting only the issue ceiling makes every
/// memory-bound kernel look like it is squandering the machine.
fn roofline_demo() {
    let (mm_ceiling, _) = unsafe { roofline::smmla_issue_ceiling(8_000_000) };
    let (dot_ceiling, _) = unsafe { roofline::sdot_issue_ceiling(8_000_000) };

    // Measured live rather than quoted from the table above: a hardcoded kernel
    // figure silently goes stale the moment the kernel changes, and a roofline
    // built on a stale numerator is worse than none.
    let (m, n, k) = (8usize, 4096usize, 4096usize);
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
    fill(&mut a, 0x9E3779B97F4A7C15);
    fill(&mut b, 0xBF58476D1CE4E5B9);
    let mut c = vec![0i32; m * n];
    let (pa, mp, kp) = pack_a_smmla(m, k, &a);
    let (pb, np) = pack_b_smmla(n, k, &b);
    let g_mm = bench_one(
        || unsafe { gemm_smmla_8x8(m, n, mp, np, kp, &pa, &pb, &mut c) }, ops, 0.8);
    let bt = pack_b_transposed(n, k, k, &b);
    let g_dot = bench_one(
        || unsafe { gemm_sdot_tiled(m, n, k, k, &a, &bt, &mut c) }, ops, 0.8);

    // Issue rates are checkable against physics in a way GOPS is not: divide by
    // ops-per-instruction and clock, and anything above a handful per cycle is
    // a deleted loop, not a fast core. The first version of this benchmark
    // reported 15.7 SMMLA/cycle and nothing in the GOPS figure gave that away.
    const CLOCK_GHZ: f64 = 3.5;
    let per_cycle = |gops: f64, ops_per_insn: f64| gops / ops_per_insn / CLOCK_GHZ;
    let mm_ipc = per_cycle(mm_ceiling, 64.0);
    let dot_ipc = per_cycle(dot_ceiling, 32.0);
    if mm_ipc > 8.0 || dot_ipc > 8.0 {
        println!(
            "\nIssue ceiling is not usable: {:.1} SMMLA/cycle, {:.1} SDOT/cycle at ~{} GHz.",
            mm_ipc, dot_ipc, CLOCK_GHZ
        );
        println!("No core issues that many; the loop was optimised out. Skipping the roofline.");
        return;
    }
    // The opposite failure, which has also happened twice: a barrier placed so
    // that it serialises the measurement chains produces a "ceiling" the real
    // kernel beats. A bound below the thing it bounds is self-refuting.
    if mm_ceiling < g_mm || dot_ceiling < g_dot {
        println!(
            "\nIssue ceiling is not usable: {:.1} GOPS against a {:.1} GOPS kernel.",
            mm_ceiling.min(dot_ceiling), g_mm
        );
        println!("A ceiling below the kernel it bounds means the measurement loop was");
        println!("serialised, not that the kernel is superhuman. Skipping the roofline.");
        return;
    }

    println!("\nSingle-core issue ceiling (register-resident, no memory traffic)");
    println!("┌──────────┬──────────────┬──────────────┬──────────┐");
    println!("│          │ ceiling GOPS │  kernel GOPS │ of peak  │");
    println!("├──────────┼──────────────┼──────────────┼──────────┤");
    println!("│ SMMLA    │ {:>12.1} │ {:>12.1} │ {:>7.1}% │",
             mm_ceiling, g_mm, g_mm / mm_ceiling * 100.0);
    println!("│ SDOT     │ {:>12.1} │ {:>12.1} │ {:>7.1}% │",
             dot_ceiling, g_dot, g_dot / dot_ceiling * 100.0);
    println!("└──────────┴──────────────┴──────────────┴──────────┘");
    println!("Kernel figures measured here at {}x{}x{}, single thread.", m, n, k);

    // The other ceiling. B for this shape is n*k = 16 MiB, so the level of the
    // hierarchy that matters is whichever one a 16 MiB footprint lands in --
    // hence the sweep rather than a single number.
    println!("\nSingle-core load bandwidth by footprint");
    println!("┌────────────┬──────────────┐");
    println!("│  footprint │        GB/s  │");
    println!("├────────────┼──────────────┤");
    let mut bw_at_b = 0.0;
    let b_bytes = n * k;
    for &bytes in &[64 << 10, 4 << 20, 16 << 20, 256 << 20] {
        let iters = ((20e9 / bytes as f64) as u64).max(1);
        let (gbs, _) = roofline::stream_bandwidth(bytes, iters);
        let label = if bytes >= (1 << 20) {
            format!("{} MiB", bytes >> 20)
        } else {
            format!("{} KiB", bytes >> 10)
        };
        let mark = if bytes == b_bytes { "  <- B panel" } else { "" };
        println!("│ {:>10} │ {:>12.1} │{}", label, gbs, mark);
        if bytes == b_bytes {
            bw_at_b = gbs;
        }
    }
    println!("└────────────┴──────────────┘");

    // A deleted benchmark loop reports an impossibly large number, not zero, and
    // this one slipped through twice before the slope gave it away. No single
    // core streams anywhere near a TB/s, so treat that as broken rather than
    // building a roofline on it and printing a confident conclusion.
    const IMPLAUSIBLE_GBS: f64 = 2000.0;
    if !bw_at_b.is_finite() || bw_at_b > IMPLAUSIBLE_GBS {
        println!(
            "\nBandwidth measurement is not usable ({:.1} GB/s at the B panel).",
            bw_at_b
        );
        println!("A single core cannot exceed ~{:.0} GB/s, so the loop was optimised", IMPLAUSIBLE_GBS);
        println!("out. Skipping the memory ceiling rather than reporting a bogus one.");
        return;
    }

    // At m = 8 with an 8-row tile there is exactly one row block, so A is read
    // once (32 KiB, negligible) and B is read once (16 MiB): intensity is
    // 2*m*n*k / (n*k) = 2*mt = 16 ops/byte. The general form when both
    // dimensions are blocked is 2*mt*nt/(mt+nt), because A is re-read per
    // column block too -- 8 ops/byte for this tile. Either way the memory
    // ceiling lands well above the issue ceiling, which is what binds.
    let intensity = 16.0;
    let mem_ceiling = bw_at_b * intensity;
    let binding = mem_ceiling.min(mm_ceiling);
    println!(
        "\nSMMLA 8x8 at m={}: intensity {:.0} ops/byte (= 2 x tile rows)",
        m, intensity
    );
    println!(
        "  memory ceiling  {:.1} GB/s x {:.0} = {:>7.1} GOPS",
        bw_at_b, intensity, mem_ceiling
    );
    println!("  issue ceiling                  = {:>7.1} GOPS", mm_ceiling);
    println!(
        "  binding ceiling {:>7.1} GOPS -> kernel is at {:.1}% of it ({})",
        binding,
        g_mm / binding * 100.0,
        if mem_ceiling < mm_ceiling { "memory-bound" } else { "issue-bound" }
    );
}

/// Show that dispatching on M wins in both regimes.
fn dispatch_demo() {
    println!("\nShape-based dispatch (SMMLA_MIN_ROWS = {})", SMMLA_MIN_ROWS);
    println!("┌──────────────────┬──────────┬──────────┬──────────┬────────────────┐");
    println!("│ M×N×K            │   SDOT   │  SMMLA   │  picked  │ vs always-SMMLA│");
    println!("├──────────────────┼──────────┼──────────┼──────────┼────────────────┤");

    for &(m, n, k) in &[(1usize, 4096usize, 4096usize), (2, 4096, 4096), (8, 4096, 4096)] {
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
        fill(&mut a, 0x9E3779B97F4A7C15);
        fill(&mut b, 0xBF58476D1CE4E5B9);
        let mut c = vec![0i32; m * n];

        let bt = pack_b_transposed(n, k, k, &b);
        let g_dot = bench_one(
            || unsafe { gemm_sdot_tiled(m, n, k, k, &a, &bt, &mut c) }, ops, 0.8);
        let (pa, mp, kp) = pack_a_smmla(m, k, &a);
        let (pb, np) = pack_b_smmla(n, k, &b);
        let g_mm = bench_one(
            || unsafe { gemm_smmla_8x8(m, n, mp, np, kp, &pa, &pb, &mut c) }, ops, 0.8);

        let picked = choose(m);
        let g_pick = if picked == Kernel::Smmla { g_mm } else { g_dot };
        // gain over an always-SMMLA implementation
        println!(
            "│ {:<16} │ {:>8.2} │ {:>8.2} │ {:>8} │ {:>13.2}× │",
            format!("{}×{}×{}", m, n, k),
            g_dot, g_mm,
            if picked == Kernel::Smmla { "SMMLA" } else { "SDOT" },
            g_pick / g_mm
        );
    }
    println!("└──────────────────┴──────────┴──────────┴──────────┴────────────────┘");
    println!("\nAn always-SMMLA implementation loses throughput at the decode step (M=1).");
    println!("Branching on M alone removes that loss - the branch costs one compare per GEMM.");

    scaling_demo();
}

/// Does the M=1 crossover survive multi-threading, or is it a single-core
/// artifact? The dispatch rule is only worth shipping if it holds at the
/// thread counts real inference uses.
fn scaling_demo() {
    let cores = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8);
    println!("\nThread scaling ({} logical cores, partitioned over columns)", cores);
    println!("┌──────────────────┬─────────┬──────────┬──────────┬──────────┐");
    println!("│ M×N×K            │ threads │   SDOT   │  SMMLA   │ SMMLA/SDOT│");
    println!("├──────────────────┼─────────┼──────────┼──────────┼──────────┤");

    for &(m, n, k) in &[(1usize, 4096usize, 4096usize), (8, 4096, 4096)] {
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
        fill(&mut a, 0x9E3779B97F4A7C15);
        fill(&mut b, 0xBF58476D1CE4E5B9);
        let mut c = vec![0i32; m * n];

        let bt = pack_b_transposed(n, k, k, &b);
        let (pa, mp, kp) = pack_a_smmla(m, k, &a);
        let (pb, np) = pack_b_smmla(n, k, &b);

        for t in [1usize, 4, 8, 16] {
            let g_dot = bench_one(
                || unsafe { gemm_sdot_mt(m, n, k, k, &a, &bt, &mut c, t) }, ops, 0.7);
            let _ = &bt;
            let g_mm = bench_one(
                || unsafe { gemm_smmla_mt(m, n, mp, np, kp, &pa, &pb, &mut c, t) }, ops, 0.7);
            println!(
                "│ {:<16} │ {:>7} │ {:>8.1} │ {:>8.1} │ {:>8.2}× │",
                if t == 1 { format!("{}×{}×{}", m, n, k) } else { String::new() },
                t, g_dot, g_mm, g_mm / g_dot
            );
        }
        println!("├──────────────────┼─────────┼──────────┼──────────┼──────────┤");
    }
    println!("└──────────────────┴─────────┴──────────┴──────────┴──────────┘");
    println!("\nIf the SMMLA/SDOT ratio stays below 1.0 at M=1 across thread counts,");
    println!("the dispatch rule is a property of the instruction, not of the core count.");
}
