//! THROWAWAY EXPERIMENT (monocp p2-threads REAL-WASI) -- do NOT commit.
//!
//! Closes the #1 gap a critic flagged in the prior worker-I/O spike
//! (monocp_worker_io_spike.rs STEP B): that spike's worker "I/O" was a MOCK
//! `record(ptr,len)` host fn appending to a `Mutex<Vec<u8>>`. No real syscall
//! ran.
//!
//! GOAL proven here: a worker OS thread, on its OWN `Store` + OWN plain
//! `wasmtime::Linker`, wired with REAL wasmtime preview1 WASI
//! (`wasmtime_wasi::p1::add_to_linker_sync` + a real `WasiP1Ctx` from
//! `WasiCtxBuilder`: a preopened temp dir + stdout captured to an in-memory
//! pipe), executes ACTUAL WASI syscalls -- a real `fd_write(stdout)` AND a real
//! `path_open`+`fd_write`+`fd_close` that CREATE AND WRITE A FILE in the
//! preopened dir -- CONCURRENTLY with the main thread driving a preview2
//! component (its lifted `emit`/`read`, canonical ABI). Then VERIFY:
//!   (a) the file exists on disk with the expected bytes (read back from host),
//!   (b) captured stdout has the expected bytes,
//!   (c) no trap (worker_entry returned 0),
//!   (d) the component's canonical-ABI state was never touched: the worker used
//!       ONLY its own plain Linker + own WasiP1Ctx; the main component instance
//!       remains callable and consistent afterwards.
//!
//! The worker core module (tests/monocp_worker_real_wasi.wasm) is a freestanding
//! C program built with wasi-sdk-27 that imports wasi_snapshot_preview1
//! {fd_write, path_open, fd_close} directly and exports `worker_entry`.

use std::sync::{Arc, Barrier, Mutex};
use wasmtime::component::{Component, Linker as CLinker};
use wasmtime::{Config, Engine, Linker, Module, Store};
use wasmtime_wasi::p1::{add_to_linker_sync, WasiP1Ctx};
use wasmtime_wasi::p2::pipe::MemoryOutputPipe;
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtxBuilder};

/// Worker core module: real WASI preview1 imports + `worker_entry` export.
const WORKER_WASM: &[u8] = include_bytes!("monocp_worker_real_wasi.wasm");

/// Reuse the prior spike's STEP_B component for the MAIN thread's concurrent
/// component work. It exposes lifted `emit(i32)` (bumps a shared counter +
/// calls a host import that many times) and `read() -> i32`. We bind its custom
/// imports with trivial host fns; we only need it to run the canonical ABI
/// concurrently so we can assert the worker never disturbs it.
const STEP_B: &[u8] = include_bytes!("stepB.wasm");

fn engine() -> Engine {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
}

#[derive(Default)]
struct CompState {
    hits: Arc<Mutex<u64>>,
}

#[test]
fn worker_thread_does_real_wasi_syscalls_concurrently() {
    let engine = engine();

    // A unique temp dir we preopen for the worker. We read the file back from
    // here on the HOST filesystem to prove the bytes really landed.
    let tmp = std::env::temp_dir().join(format!(
        "monocp_worker_real_wasi_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let expected_file = tmp.join("worker_out.txt");
    // Make sure it doesn't exist yet -- the worker must CREATE it.
    assert!(
        !expected_file.exists(),
        "precondition: target file must not exist before the worker runs"
    );

    // ---- MAIN thread: a preview2 component on its own Store + component Linker.
    let component = Component::new(&engine, STEP_B).unwrap();
    let mut cstore = Store::new(&engine, CompState::default());
    let mut clinker: CLinker<CompState> = CLinker::new(&engine);
    clinker
        .instance("monocp:threads/threading")
        .unwrap()
        .func_wrap(
            "thread-spawn",
            |_c: wasmtime::StoreContextMut<'_, CompState>, (a,): (i32,)| Ok((a,)),
        )
        .unwrap();
    // The component's `record` host import -- count invocations so we can prove
    // the component's canonical-ABI path ran fully and consistently.
    clinker
        .instance("monocp:io/host")
        .unwrap()
        .func_wrap(
            "record",
            |c: wasmtime::StoreContextMut<'_, CompState>, (_p, _l): (i32, i32)| {
                *c.data().hits.lock().unwrap() += 1;
                Ok(())
            },
        )
        .unwrap();
    let cinstance = clinker.instantiate(&mut cstore, &component).unwrap();

    // ---- WORKER thread: OWN Store<WasiP1Ctx> + OWN plain Linker with REAL WASI.
    let stdout_pipe = MemoryOutputPipe::new(64 * 1024);
    let stderr_pipe = MemoryOutputPipe::new(64 * 1024);

    // Barrier so the worker's hot loop of REAL syscalls and the main thread's
    // component canonical-ABI loop are provably in-flight at the same time
    // (not run sequentially).
    let barrier = Arc::new(Barrier::new(2));

    let weng = engine.clone();
    let wtmp = tmp.clone();
    let wstdout = stdout_pipe.clone();
    let wstderr = stderr_pipe.clone();
    let wbarrier = barrier.clone();
    // Number of extra real fd_write syscalls the worker does in a busy loop.
    const WORKER_LOOPS: i32 = 20_000;
    let worker = std::thread::spawn(move || -> i32 {
        // Real preview1 WASI ctx: preopen the temp dir as guest "/" (fd 3), and
        // capture stdout/stderr to in-memory pipes.
        let wasi = WasiCtxBuilder::new()
            .stdout(wstdout)
            .stderr(wstderr)
            .preopened_dir(&wtmp, "/", DirPerms::all(), FilePerms::all())
            .unwrap()
            .build_p1();

        let mut wstore = Store::new(&weng, wasi);

        // OWN plain core Linker -- real preview1 handlers wired here.
        let mut wlinker: Linker<WasiP1Ctx> = Linker::new(&weng);
        add_to_linker_sync(&mut wlinker, |t| t).unwrap();

        let wmod = Module::new(&weng, WORKER_WASM).unwrap();
        let winst = wlinker.instantiate(&mut wstore, &wmod).unwrap();

        let entry = winst
            .get_typed_func::<(i32,), (i32,)>(&mut wstore, "worker_entry")
            .unwrap();
        // Release at the same instant as the main thread.
        wbarrier.wait();
        // REAL syscalls happen inside this call: fd_write(stdout) + a hot loop
        // of fd_write(stderr) + path_open + fd_write(file) + fd_close.
        let (rc,) = entry.call(&mut wstore, (WORKER_LOOPS,)).unwrap();
        rc
    });

    // ---- MAIN thread proceeds CONCURRENTLY: drive the component canonical ABI.
    const ITERS: i32 = 4000;
    let emit = cinstance
        .get_typed_func::<(i32,), ()>(&mut cstore, "emit")
        .unwrap();
    // Start the component's lifted loop at the same instant the worker starts
    // its real-syscall loop.
    barrier.wait();
    emit.call(&mut cstore, (ITERS,)).unwrap();
    emit.post_return(&mut cstore).unwrap();

    // Join the worker; capture its return code (0 == every WASI syscall ok).
    let rc = worker.join().unwrap();

    // ---- VERIFY (c): no trap, worker_entry returned 0.
    eprintln!("worker_entry returned rc={rc} (0 == all real WASI syscalls succeeded)");
    assert_eq!(
        rc, 0,
        "worker_entry must return 0; non-zero encodes which WASI step failed \
         (1xxx stdout fd_write, 2xxx path_open, 3xxx file fd_write, 4xxx fd_close)"
    );

    // ---- VERIFY (a): the file exists on the HOST filesystem with exact bytes.
    let on_disk = std::fs::read(&expected_file)
        .unwrap_or_else(|e| panic!("worker file not on disk at {expected_file:?}: {e}"));
    eprintln!(
        "host filesystem: {expected_file:?} = {:?}",
        String::from_utf8_lossy(&on_disk)
    );
    assert_eq!(
        on_disk, b"WORKER-FILE-PAYLOAD-12345",
        "file bytes written by the worker's REAL path_open+fd_write must match"
    );

    // ---- VERIFY (b): captured stdout has the exact bytes.
    let out = stdout_pipe.contents();
    eprintln!("captured worker stdout = {:?}", String::from_utf8_lossy(&out));
    assert_eq!(
        &out[..],
        b"WORKER-STDOUT-HELLO\n",
        "captured stdout from the worker's REAL fd_write must match"
    );

    // The hot loop wrote WORKER_LOOPS real fd_write(stderr) syscalls; the
    // captured stderr pipe must hold exactly that many '.' bytes -- direct
    // evidence the worker really executed that many WASI syscalls.
    let err = stderr_pipe.contents();
    let dots = err.iter().filter(|&&b| b == b'.').count();
    eprintln!(
        "captured worker stderr: {dots} '.' bytes from {WORKER_LOOPS} real fd_write syscalls"
    );
    assert_eq!(
        dots, WORKER_LOOPS as usize,
        "every real fd_write(stderr) syscall in the worker hot loop must have landed"
    );

    // ---- VERIFY (d): the component's canonical-ABI state is intact & untouched.
    // The component ran its full lifted path concurrently; its host import was
    // hit ITERS times, and a fresh lifted `read()` still works (instance not
    // corrupted by the worker, which used only its own Linker + own WasiP1Ctx).
    let hits = *cstore.data().hits.lock().unwrap();
    let read = cinstance
        .get_typed_func::<(), (i32,)>(&mut cstore, "read")
        .unwrap();
    let counter = read.call(&mut cstore, ()).unwrap().0;
    read.post_return(&mut cstore).unwrap();
    eprintln!(
        "component canonical-ABI intact: record host-import hits={hits} (expected {ITERS}), \
         lifted read() counter={counter}"
    );
    assert_eq!(
        hits, ITERS as u64,
        "component's canonical-ABI path ran fully and was not disturbed by the worker"
    );
    assert!(
        counter >= ITERS,
        "lifted read() still callable after worker ran -> component instance not corrupted"
    );

    eprintln!(
        "DELIVERABLE: a worker OS thread executed REAL WASI syscalls (fd_write to \
         stdout + path_open/fd_write/fd_close creating a file) CONCURRENTLY with the \
         main thread's preview2 component, on its OWN Store + OWN plain Linker + OWN \
         real WasiP1Ctx. Bytes verified on disk AND in captured stdout."
    );

    // Best-effort cleanup of the throwaway temp dir.
    let _ = std::fs::remove_dir_all(&tmp);
}
