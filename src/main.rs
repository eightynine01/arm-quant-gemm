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

    crossover_demo();
    dispatch_demo();
    roofline_demo();
    unroll_demo();
    thread_balance_demo();
    engine_demo();
}

/// The one-call path, and a check that it actually picks what it claims.
///
/// An API that encodes measured rules is only worth shipping if it produces the
/// same answer as the kernels it dispatches to, so every shape here is compared
/// against the scalar reference rather than against the fast path.
fn engine_demo() {
    println!("\nEngine: one call, both rules applied");
    println!("┌──────────────────┬─────────┬────────────┬──────────┬─────────┐");
    println!("│ M×N×K            │ threads │  picked    │   GOPS   │ correct │");
    println!("├──────────────────┼─────────┼────────────┼──────────┼─────────┤");

    for &(m, n, k, nt) in &[
        (1usize, 1024usize, 1024usize, 8usize),
        (8, 1024, 1024, 8),
        (8, 4096, 4096, 12),
        (8, 4096, 4096, 16),
        (64, 1024, 1024, 12),
    ] {
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
        fill(&mut a, 0x9E3779B97F4A7C15);
        fill(&mut b, 0xBF58476D1CE4E5B9);
        let mut c_ref = vec![0i32; m * n];
        gemm_scalar(m, n, k, &a, &b, &mut c_ref);

        let bt = pack_b_transposed(n, k, k, &b);
        let (pa, mp, kp) = pack_a_smmla(m, k, &a);
        let (pb, np) = pack_b_smmla(n, k, &b);
        let mut c = vec![0i32; m * n];

        let eng = Engine::new(nt);
        unsafe { eng.gemm(m, n, k, mp, np, kp, &a, &bt, &pa, &pb, &mut c) };
        let ok = c == c_ref;

        let g = bench_one(
            || unsafe { eng.gemm(m, n, k, mp, np, kp, &a, &bt, &pa, &pb, &mut c) },
            ops, 0.5);
        let picked = match choose(m) {
            Kernel::Smmla => "SMMLA",
            Kernel::Sdot => "SDOT",
        };
        let barrier = if nt <= spin_threshold() { "spin" } else { "chan" };
        println!(
            "│ {:<16} │ {:>7} │ {:<10} │ {:>8.1} │ {:>7} │",
            format!("{}×{}×{}", m, n, k), nt,
            format!("{}/{}", picked, barrier), g,
            if ok { "match" } else { "MISMATCH" }
        );
    }
    println!("└──────────────────┴─────────┴────────────┴──────────┴─────────┘");
    println!("SMMLA at M>={}, SDOT below. Spin barrier up to {} threads ({} cores).",
             SMMLA_MIN_ROWS, spin_threshold(),
             if performance_cores().is_some() { "detected" } else { "measured default" });
}

/// Why does 16 threads lose to 8?
///
/// The scaling table shows throughput *falling* from 8 threads to 16 on a
/// machine with 16 performance cores, which is the kind of anomaly that has to
/// be explained rather than reported. Two candidates: work landing on the 8
/// efficiency cores, or contention once both performance clusters are busy.
///
/// They predict different things about the *spread* of per-thread times, so
/// timing each worker separates them. Efficiency-core placement gives a few
/// stragglers several times slower than the rest; contention slows everyone
/// roughly equally.
fn thread_balance_demo() {
    let (m, n, k) = (8usize, 4096usize, 4096usize);
    let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
    fill(&mut a, 0x9E3779B97F4A7C15);
    fill(&mut b, 0xBF58476D1CE4E5B9);
    let (pa, mp, kp) = pack_a_smmla(m, k, &a);
    let (pb, np) = pack_b_smmla(n, k, &b);
    let mut c = vec![0i32; m * n];

    println!("\nPer-thread time at {}x{}x{} (equal column chunks)", m, n, k);
    println!("┌─────────┬──────────┬──────────┬──────────┬──────────┬────────┐");
    println!("│ threads │  fastest │  slowest │ slow/fast│ wall  ms │  GOPS  │");
    println!("├─────────┼──────────┼──────────┼──────────┼──────────┼────────┤");

    for &nt in &[8usize, 12, 16, 20, 24] {
        let colpairs = np / 2;
        let cp_chunk = colpairs.div_ceil(nt);
        let times = std::sync::Mutex::new(Vec::<f64>::new());
        let cp = c.as_mut_ptr() as usize;
        let clen = c.len();

        let t0 = Instant::now();
        std::thread::scope(|s| {
            for t in 0..nt {
                let cp0 = t * cp_chunk;
                if cp0 >= colpairs {
                    break;
                }
                let cp1 = ((t + 1) * cp_chunk).min(colpairs);
                let times = &times;
                let (pa, pb) = (&pa, &pb);
                s.spawn(move || {
                    let t1 = Instant::now();
                    unsafe {
                        let cs = std::slice::from_raw_parts_mut(cp as *mut i32, clen);
                        smmla_8x8_colrange(m, n, mp, kp, pa, pb, cs, cp0, cp1);
                    }
                    times.lock().unwrap().push(t1.elapsed().as_secs_f64());
                });
            }
        });
        let wall = t0.elapsed().as_secs_f64();

        let mut v = times.into_inner().unwrap();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let (fast, slow) = (v[0], v[v.len() - 1]);
        let gops = 2.0 * m as f64 * n as f64 * k as f64 / wall / 1e9;
        println!(
            "│ {:>7} │ {:>7.2}ms │ {:>7.2}ms │ {:>7.2}× │ {:>7.2}  │ {:>6.1} │",
            nt, fast * 1e3, slow * 1e3, slow / fast, wall * 1e3, gops
        );
    }
    println!("└─────────┴──────────┴──────────┴──────────┴──────────┴────────┘");

    // The gap between the slowest worker and the wall clock is not compute, so
    // measure what it is: the same scope with empty bodies.
    println!("\nSpawn/join cost alone (identical scope, empty bodies)");
    println!("┌─────────┬───────────┐");
    println!("│ threads │  overhead │");
    println!("├─────────┼───────────┤");
    for &nt in &[8usize, 12, 16, 20, 24] {
        let reps = 200;
        let t0 = Instant::now();
        for _ in 0..reps {
            std::thread::scope(|s| {
                for _ in 0..nt {
                    s.spawn(|| {});
                }
            });
        }
        println!("│ {:>7} │ {:>7.2}ms │", nt, t0.elapsed().as_secs_f64() / reps as f64 * 1e3);
    }
    println!("└─────────┴───────────┘");
    println!("16 performance cores + 8 efficiency cores on this machine.");

    // Does a persistent pool recover it?
    println!("\nPersistent pool vs per-call spawn");
    println!("┌─────────┬───────────┬───────────┬─────────┬─────────┐");
    println!("│ threads │ spawn GOPS│  pool GOPS│  gain   │ correct │");
    println!("├─────────┼───────────┼───────────┼─────────┼─────────┤");
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    for &nt in &[4usize, 8, 12, 16, 24] {
        let pool = Pool::new(nt);
        let mut c1 = vec![0i32; m * n];
        let mut c2 = vec![0i32; m * n];
        unsafe { gemm_smmla_mt(m, n, mp, np, kp, &pa, &pb, &mut c1, nt) };
        unsafe { gemm_smmla_pool(m, n, mp, np, kp, &pa, &pb, &mut c2, &pool) };
        let ok = c1 == c2;

        let g_spawn = bench_one(
            || unsafe { gemm_smmla_mt(m, n, mp, np, kp, &pa, &pb, &mut c1, nt) }, ops, 0.8);
        let g_pool = bench_one(
            || unsafe { gemm_smmla_pool(m, n, mp, np, kp, &pa, &pb, &mut c2, &pool) }, ops, 0.8);
        println!(
            "│ {:>7} │ {:>9.1} │ {:>9.1} │ {:>6.2}× │ {:>7} │",
            nt, g_spawn, g_pool, g_pool / g_spawn,
            if ok { "match" } else { "MISMATCH" }
        );
    }
    println!("└─────────┴───────────┴───────────┴─────────┴─────────┘");

    // The dispatch claim was measured with both sides paying per-call thread
    // creation. If that cost dominated, the ratio was partly a measurement of
    // thread management. Re-run it on equal, pooled footing.
    println!("\nDispatch rule re-measured on the pool (spawn cost removed)");
    println!("┌──────────────────┬─────────┬──────────┬──────────┬───────────┐");
    println!("│ M×N×K            │ threads │   SDOT   │  SMMLA   │ SMMLA/SDOT│");
    println!("├──────────────────┼─────────┼──────────┼──────────┼───────────┤");
    for &(mm, nn, kk) in &[(1usize, 4096usize, 4096usize), (8, 4096, 4096)] {
        let ops2 = 2.0 * mm as f64 * nn as f64 * kk as f64;
        let (mut a2, mut b2) = (vec![0i8; mm * kk], vec![0i8; kk * nn]);
        fill(&mut a2, 0x9E3779B97F4A7C15);
        fill(&mut b2, 0xBF58476D1CE4E5B9);
        let bt2 = pack_b_transposed(nn, kk, kk, &b2);
        let (pa2, mp2, kp2) = pack_a_smmla(mm, kk, &a2);
        let (pb2, np2) = pack_b_smmla(nn, kk, &b2);
        let mut cd = vec![0i32; mm * nn];
        let mut cs = vec![0i32; mm * nn];
        for &nt in &[1usize, 4, 8, 12, 16] {
            let pool = Pool::new(nt);
            let gd = bench_one(
                || unsafe { gemm_sdot_pool(mm, nn, kk, kk, &a2, &bt2, &mut cd, &pool) },
                ops2, 0.6);
            let gs = bench_one(
                || unsafe { gemm_smmla_pool(mm, nn, mp2, np2, kp2, &pa2, &pb2, &mut cs, &pool) },
                ops2, 0.6);
            let label = if nt == 1 { format!("{}×{}×{}", mm, nn, kk) } else { String::new() };
            println!("│ {:<16} │ {:>7} │ {:>8.1} │ {:>8.1} │ {:>8.2}× │",
                     label, nt, gd, gs, gs / gd);
        }
        println!("├──────────────────┼─────────┼──────────┼──────────┼───────────┤");
    }
    println!("└──────────────────┴─────────┴──────────┴──────────┴───────────┘");
    println!("If SMMLA/SDOT stays below 1.0 at M=1 here too, the dispatch rule");
    println!("is a property of the instructions and not of the thread harness.");

    // The pooled path reaches 45% of 12 single-core issue ceilings. Two
    // candidates for the rest, and they predict different things:
    //   pool dispatch  -> shows up with empty bodies, independent of shape
    //   shared memory  -> vanishes when the B panel fits in cache
    println!("\nWhat bounds the pooled path?");
    let pool12 = Pool::new(12);
    let reps = 2000;
    let t0 = Instant::now();
    for _ in 0..reps {
        pool12.run(|_| {});
    }
    let dispatch = t0.elapsed().as_secs_f64() / reps as f64;
    println!("  pool dispatch, empty bodies, 12 workers : {:>8.3}ms", dispatch * 1e3);

    // Same op count, B panel 16 MiB vs 1 MiB.
    for &(mm, nn, kk, tag) in &[
        (8usize, 4096usize, 4096usize, "B = 16 MiB"),
        (128, 1024, 1024, "B =  1 MiB"),
    ] {
        let ops2 = 2.0 * mm as f64 * nn as f64 * kk as f64;
        let (mut a2, mut b2) = (vec![0i8; mm * kk], vec![0i8; kk * nn]);
        fill(&mut a2, 0x9E3779B97F4A7C15);
        fill(&mut b2, 0xBF58476D1CE4E5B9);
        let (pa2, mp2, kp2) = pack_a_smmla(mm, kk, &a2);
        let (pb2, np2) = pack_b_smmla(nn, kk, &b2);
        let mut c2 = vec![0i32; mm * nn];
        let g = bench_one(
            || unsafe { gemm_smmla_pool(mm, nn, mp2, np2, kp2, &pa2, &pb2, &mut c2, &pool12) },
            ops2, 0.8);
        let per_call = ops2 / (g * 1e9);
        println!("  {} {:>4}×{}×{}: {:>7.1} GOPS, {:>6.3}ms/call, dispatch = {:>4.1}%",
                 tag, mm, nn, kk, g, per_call * 1e3, dispatch / per_call * 100.0);
    }

    // Replace the channels with a spin barrier and re-measure.
    println!("\nSpin barrier vs channel pool");
    println!("┌─────────┬───────────┬───────────┬─────────┬─────────┐");
    println!("│ threads │ chan GOPS │ spin GOPS │  gain   │ correct │");
    println!("├─────────┼───────────┼───────────┼─────────┼─────────┤");
    let ops3 = 2.0 * m as f64 * n as f64 * k as f64;
    // Repeated because a single reading at 16 threads once showed spin losing
    // 0.76x, and that number was written into the README and into
    // SPIN_MAX_THREADS before it was checked. Five further runs all showed spin
    // winning (1.28x-1.82x). Near saturation the run-to-run spread is wide
    // enough that one sample decides nothing.
    for &nt in &[8usize, 12, 16, 20, 24] {
        let mut c1 = vec![0i32; m * n];
        let mut c2 = vec![0i32; m * n];
        // The two pools are built and dropped in separate scopes on purpose.
        // Holding both alive at once let the spin workers burn cores while the
        // channel pool was being timed, which depressed the channel numbers by
        // ~25% and inflated the ratio. Idle spinners are not free.
        let gc = {
            let cpool = Pool::new(nt);
            unsafe { gemm_smmla_pool(m, n, mp, np, kp, &pa, &pb, &mut c1, &cpool) };
            bench_one(
                || unsafe { gemm_smmla_pool(m, n, mp, np, kp, &pa, &pb, &mut c1, &cpool) },
                ops3, 0.8)
        };
        let gs = {
            let spool = SpinPool::new(nt);
            unsafe { gemm_smmla_spin(m, n, mp, np, kp, &pa, &pb, &mut c2, &spool) };
            bench_one(
                || unsafe { gemm_smmla_spin(m, n, mp, np, kp, &pa, &pb, &mut c2, &spool) },
                ops3, 0.8)
        };
        let ok = c1 == c2;
        println!("│ {:>7} │ {:>9.1} │ {:>9.1} │ {:>6.2}× │ {:>7} │",
                 nt, gc, gs, gs / gc, if ok { "match" } else { "MISMATCH" });
    }
    println!("└─────────┴───────────┴───────────┴─────────┴─────────┘");
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

/// Where is the crossover, actually?
///
/// `SMMLA_MIN_ROWS = 2` follows from the tile geometry — `SMMLA` emits two rows,
/// so M=1 wastes half and M>=2 does not — and the benchmarks above only ever run
/// M=1 and M=8. A boundary asserted at the value theory predicts, with no
/// measurement on either side of it, is the same mistake that put the spin
/// cliff at 12 instead of 20.
fn crossover_demo() {
    println!("\nWhere the dispatch crossover actually is (1 thread, N=K=4096)");
    println!("┌─────┬──────────┬──────────┬───────────┬──────────┐");
    println!("│  M  │   SDOT   │  SMMLA   │ SMMLA/SDOT│  picked  │");
    println!("├─────┼──────────┼──────────┼───────────┼──────────┤");
    for m in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16] {
        let (n, k) = (4096usize, 4096usize);
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let (mut a, mut b) = (vec![0i8; m * k], vec![0i8; k * n]);
        fill(&mut a, 0x9E3779B97F4A7C15);
        fill(&mut b, 0xBF58476D1CE4E5B9);
        let bt = pack_b_transposed(n, k, k, &b);
        let (pa, mp, kp) = pack_a_smmla(m, k, &a);
        let (pb, np) = pack_b_smmla(n, k, &b);
        let mut c = vec![0i32; m * n];
        let gd = bench_one(
            || unsafe { gemm_sdot_tiled(m, n, k, k, &a, &bt, &mut c) }, ops, 0.6);
        let gs = bench_one(
            || unsafe { gemm_smmla_8x8(m, n, mp, np, kp, &pa, &pb, &mut c) }, ops, 0.6);
        let picked = if choose(m) == Kernel::Smmla { "SMMLA" } else { "SDOT" };
        let flag = if (gs > gd) != (choose(m) == Kernel::Smmla) { "  <-- rule disagrees" } else { "" };
        println!("│ {:>3} │ {:>8.1} │ {:>8.1} │ {:>8.2}× │ {:<8} │{}",
                 m, gd, gs, gs / gd, picked, flag);
    }
    println!("└─────┴──────────┴──────────┴───────────┴──────────┘");
    println!("The rule should pick SMMLA exactly when the ratio is above 1.00.");
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
