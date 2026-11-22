//! THROWAWAY EXPERIMENT (monocp p2-threads-io) -- do NOT commit.
//!
//! Question: can a worker OS thread perform real I/O safely while the main
//! thread drives a preview2 component, and HOW?
//!
//! Builds on exp-4 / p4 (monocp_threads_full_flow.rs), which proved a worker OS
//! thread can re-instantiate an inner core module on its OWN Store sharing the
//! provider's EXPORTED shared memory and run wasi_thread_start with correct
//! atomics. That spike's worker only touched raw memory + a thread-spawn STUB.
//!
//! Here we go further:
//!   STEP A -- DEMONSTRATE THE FAILURE: show what happens when something crosses
//!             the component boundary (a lifted/canonical-ABI call) while the
//!             instance is already "entered". Expect CannotEnterComponent.
//!   STEP B -- THE FIX: a worker re-instantiates the inner core module on its
//!             OWN Store+Linker, binding plain WASI-style host imports (a
//!             `record(ptr,len)` host fn) to thread-safe host state behind a
//!             Mutex, and does real "I/O" CONCURRENTLY with main host I/O while
//!             both touch the shared linear memory. The worker NEVER touches the
//!             component instance state.
//!   STEP C -- characterize: what the worker CAN reach (its own plain imports)
//!             vs CANNOT (the component's preview2 custom wit imports).

use std::sync::{Arc, Mutex};
use wasmtime::component::{Component, Instance as CInstance, Linker as CLinker};
use wasmtime::{Config, Engine, Linker, Module, SharedMemory, Store};

fn engine() -> Engine {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
}

// ---------------------------------------------------------------------------
// STEP A: prove the boundary cannot be crossed while the instance is entered.
// ---------------------------------------------------------------------------
//
// `stepA.wasm`: lifted export `run` increments a counter then calls host import
// `notify`. Inside the host `notify` callback -- i.e. while the component is
// ENTERED and mid canonical-ABI flush -- we attempt to call back into another
// lifted export (`read`) on the SAME instance. That is precisely the canonical-
// ABI state a worker crossing the boundary concurrently would corrupt. Wasmtime
// guards this with the reentrance check; we capture the EXACT result.
const STEP_A: &[u8] = include_bytes!("stepA.wasm");

struct StateA {
    // filled by the test before the call so the host callback can reach the
    // component instance + a typed `read` func to attempt re-entry.
    instance: Option<CInstance>,
    reentry_result: Option<Result<i32, String>>,
}

#[test]
fn step_a_reentry_into_entered_instance_traps() {
    let engine = engine();
    let component = Component::new(&engine, STEP_A).unwrap();
    let mut store = Store::new(
        &engine,
        StateA {
            instance: None,
            reentry_result: None,
        },
    );
    let mut linker: CLinker<StateA> = CLinker::new(&engine);
    linker
        .instance("monocp:io/host")
        .unwrap()
        .func_wrap(
            "notify",
            |mut caller: wasmtime::StoreContextMut<'_, StateA>, (): ()| {
                // We are INSIDE a lifted call (`run`). Attempt to cross the
                // boundary AGAIN by calling the lifted `read` on the same
                // instance -- exactly what a concurrent worker would do.
                let inst = caller.data().instance.expect("instance set");
                let res: Result<i32, String> = (|| {
                    let read = inst
                        .get_typed_func::<(), (i32,)>(&mut caller, "read")
                        .map_err(|e| format!("lookup: {e}"))?;
                    let v = read
                        .call(&mut caller, ())
                        .map_err(|e| format!("call: {e}"))?
                        .0;
                    read.post_return(&mut caller)
                        .map_err(|e| format!("post_return: {e}"))?;
                    Ok(v)
                })();
                caller.data_mut().reentry_result = Some(res);
                Ok(())
            },
        )
        .unwrap();

    let instance = linker.instantiate(&mut store, &component).unwrap();
    store.data_mut().instance = Some(instance);

    let run = instance
        .get_typed_func::<(), ()>(&mut store, "run")
        .unwrap();
    run.call(&mut store, ()).unwrap();
    run.post_return(&mut store).unwrap();

    let reentry = store.data_mut().reentry_result.take().expect("ran");
    match &reentry {
        Ok(v) => eprintln!(
            "STEP A: SURPRISE -- reentrant lifted read SUCCEEDED, returned {v}"
        ),
        Err(e) => eprintln!("STEP A: reentrant lifted read FAILED as expected: {e}"),
    }
    assert!(
        reentry.is_err(),
        "crossing the component boundary while ENTERED must be rejected"
    );
    let msg = reentry.unwrap_err();
    assert!(
        msg.contains("cannot reenter")
            || msg.to_lowercase().contains("reenter")
            || msg.to_lowercase().contains("enter"),
        "expected a reentrance/CannotEnterComponent error, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// STEP B: THE FIX -- worker does real host I/O via its OWN plain imports.
// ---------------------------------------------------------------------------
const STEP_B: &[u8] = include_bytes!("stepB.wasm");

/// Thread-safe host "I/O sink" shared between the MAIN component store's host
/// import and the WORKER store's host import. Mirrors the monocp preview1 host
/// pattern (shared state behind a Mutex) + wasmtime-wasi-threads instantiate-
/// per-thread.
#[derive(Default, Clone)]
struct IoSink(Arc<Mutex<Vec<u8>>>);

/// State carried by BOTH the component store and the worker store. The worker's
/// Linker binds the SAME `record` host import to the SAME `IoSink`.
struct HostState {
    sink: IoSink,
    mem: Option<SharedMemory>,
}

#[test]
fn step_b_worker_real_io_via_own_imports() {
    let engine = engine();
    let component = Component::new(&engine, STEP_B).unwrap();

    let sink = IoSink::default();

    // ---- main: instantiate the component; bind `record` host import. -------
    let mut store = Store::new(
        &engine,
        HostState {
            sink: sink.clone(),
            mem: None,
        },
    );
    let mut linker: CLinker<HostState> = CLinker::new(&engine);
    // thread-spawn stub (component custom wit import)
    linker
        .instance("monocp:threads/threading")
        .unwrap()
        .func_wrap(
            "thread-spawn",
            |_c: wasmtime::StoreContextMut<'_, HostState>, (a,): (i32,)| Ok((a,)),
        )
        .unwrap();
    // `record(ptr,len)` host import -- a real fd_write-shaped sink. The lifted
    // path reads `len` bytes at `ptr` from the inner core memory and appends to
    // the shared mutex-protected host buffer.
    linker
        .instance("monocp:io/host")
        .unwrap()
        .func_wrap(
            "record",
            |mut caller: wasmtime::StoreContextMut<'_, HostState>, (ptr, len): (i32, i32)| {
                let mem = caller.data().mem.clone().expect("mem set");
                let data = mem.data();
                let start = ptr as usize;
                let end = start + len as usize;
                // SAFETY: shared memory backing; we only read.
                let bytes: Vec<u8> = unsafe {
                    (start..end).map(|i| *(data[i].get())).collect()
                };
                caller.data().sink.0.lock().unwrap().extend_from_slice(&bytes);
                Ok(())
            },
        )
        .unwrap();

    let instance = linker.instantiate(&mut store, &component).unwrap();

    // Provider core instance index 0 EXPORTS the shared memory; worker module
    // is core instance index ... probe both, like p4.
    let mut shared: Option<SharedMemory> = None;
    for idx in 0..6u32 {
        let got = instance.get_core_shared_memory(&mut store, idx, "memory");
        eprintln!(
            "STEP B probe: core idx {idx} shared mem = {}",
            got.is_some()
        );
        if shared.is_none() {
            shared = got;
        }
    }
    let shared = shared.expect("provider shared memory reachable");
    store.data_mut().mem = Some(shared.clone());

    // Find which core instance is the WORKER module (imports memory + record).
    let mut worker_mod: Option<Module> = None;
    let mut worker_idx = u32::MAX;
    for idx in 0..6u32 {
        if let Some(m) = instance.get_core_instance_module(&mut store, idx) {
            let imports_mem = m.imports().any(|i| i.ty().memory().is_some());
            let imports_record = m.imports().any(|i| i.name() == "record");
            eprintln!(
                "STEP B probe: core idx {idx} module imports_mem={imports_mem} imports_record={imports_record} imports={:?}",
                m.imports().map(|i| format!("{}::{}", i.module(), i.name())).collect::<Vec<_>>()
            );
            if imports_mem && imports_record && worker_mod.is_none() {
                worker_mod = Some(m);
                worker_idx = idx;
            }
        }
    }
    let worker_mod = worker_mod.expect("worker module (imports mem+record) found");
    eprintln!("STEP B: worker module is core idx {worker_idx}");

    // Pre-seed two distinct byte patterns into the shared memory at known
    // offsets so we can tell main's I/O from the worker's I/O.
    const MAIN_OFF: i32 = 100;
    const WORK_OFF: i32 = 200;
    const N: usize = 8;
    {
        let d = shared.data();
        for i in 0..N {
            unsafe {
                *(d[MAIN_OFF as usize + i].get()) = b'M';
                *(d[WORK_OFF as usize + i].get()) = b'W';
            }
        }
    }

    // ---- worker OS thread: OWN Store + OWN Linker (plain wasmtime::Linker on
    // core modules), binds `record` to the SAME IoSink, shares provider memory.
    let weng = engine.clone();
    let wmod = worker_mod.clone();
    let wmem = shared.clone();
    let wsink = sink.clone();
    let handle = std::thread::spawn(move || {
        let mut wstore = Store::new(
            &weng,
            HostState {
                sink: wsink,
                mem: Some(wmem.clone()),
            },
        );
        let mut wlinker: Linker<HostState> = Linker::new(&weng);
        // import "env" "memory" -> shared provider memory
        wlinker.define(&mut wstore, "env", "memory", wmem.clone()).unwrap();
        // import "host" "record" -> PLAIN thread-safe host fn (own import!)
        wlinker
            .func_wrap(
                "host",
                "record",
                |mut caller: wasmtime::Caller<'_, HostState>, ptr: i32, len: i32| {
                    let mem = caller.data().mem.clone().unwrap();
                    let data = mem.data();
                    let bytes: Vec<u8> = unsafe {
                        (ptr as usize..(ptr + len) as usize)
                            .map(|i| *(data[i].get()))
                            .collect()
                    };
                    caller.data().sink.0.lock().unwrap().extend_from_slice(&bytes);
                },
            )
            .unwrap();
        // import "wasi" "thread-spawn" -> stub (i32)->i32
        wlinker
            .func_wrap("wasi", "thread-spawn", |_c: wasmtime::Caller<'_, HostState>, a: i32| a)
            .unwrap();

        let winst = wlinker.instantiate(&mut wstore, &wmod).unwrap();
        // worker entry: write WORK_OFF..+N to the sink many times + bump counter
        let start = winst
            .get_typed_func::<(i32, i32), ()>(&mut wstore, "wasi_thread_start")
            .unwrap();
        // arg encodes (offset<<16)|count via the module; here we just pass count
        start.call(&mut wstore, (1, ITERS)).unwrap();
    });

    // ---- main thread: drive the component's lifted `emit` CONCURRENTLY, which
    // bumps the shared counter and calls the `record` host import ITERS times.
    const ITERS: i32 = 5000;
    let emit = instance
        .get_typed_func::<(i32,), ()>(&mut store, "emit")
        .unwrap();
    emit.call(&mut store, (ITERS,)).unwrap();
    emit.post_return(&mut store).unwrap();

    handle.join().unwrap();

    // counter: both threads atomically bumped addr 0 ITERS times each.
    let read = instance
        .get_typed_func::<(), (i32,)>(&mut store, "read")
        .unwrap();
    let counter = read.call(&mut store, ()).unwrap().0;
    read.post_return(&mut store).unwrap();

    // I/O: the sink must contain BOTH 'M' bytes (main, via component lifted path)
    // and 'W' bytes (worker, via its OWN plain import). Count them.
    let buf = sink.0.lock().unwrap();
    let m_count = buf.iter().filter(|&&b| b == b'M').count();
    let w_count = buf.iter().filter(|&&b| b == b'W').count();
    eprintln!(
        "STEP B: counter={counter} (expected {}), sink total={} bytes, 'M'={m_count} 'W'={w_count}",
        2 * ITERS,
        buf.len()
    );
    assert_eq!(counter, 2 * ITERS, "atomic counter: no lost updates across both threads");
    assert_eq!(
        m_count,
        ITERS as usize * N,
        "main thread host I/O via component lifted path landed in full"
    );
    assert_eq!(
        w_count,
        ITERS as usize * N,
        "worker thread host I/O via its OWN plain import landed in full"
    );
}

// ---------------------------------------------------------------------------
// STEP C: confirm the worker CANNOT reach the component's lifted/custom-wit
// path (only its plain core imports). We assert the worker module has NO way to
// invoke the component's lifted `emit`/`read` -- those are component-level
// exports, not core exports of the worker module.
// ---------------------------------------------------------------------------
#[test]
fn step_c_worker_module_has_no_lifted_exports() {
    let engine = engine();
    let component = Component::new(&engine, STEP_B).unwrap();
    let mut store = Store::new(
        &engine,
        HostState {
            sink: IoSink::default(),
            mem: None,
        },
    );
    let mut linker: CLinker<HostState> = CLinker::new(&engine);
    linker
        .instance("monocp:threads/threading")
        .unwrap()
        .func_wrap("thread-spawn", |_c: wasmtime::StoreContextMut<'_, HostState>, (a,): (i32,)| Ok((a,)))
        .unwrap();
    linker
        .instance("monocp:io/host")
        .unwrap()
        .func_wrap("record", |_c: wasmtime::StoreContextMut<'_, HostState>, (_p, _l): (i32, i32)| Ok(()))
        .unwrap();
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // worker module = core idx with mem+record imports
    let mut worker_mod = None;
    for idx in 0..6u32 {
        if let Some(m) = instance.get_core_instance_module(&mut store, idx) {
            if m.imports().any(|i| i.ty().memory().is_some())
                && m.imports().any(|i| i.name() == "record")
            {
                worker_mod = Some(m);
            }
        }
    }
    let worker_mod = worker_mod.unwrap();
    let exports: Vec<String> = worker_mod.exports().map(|e| e.name().to_string()).collect();
    eprintln!("STEP C: worker core module exports = {exports:?}");
    // The worker has plain core funcs (record-driving entry) but NO canonical-
    // ABI realloc/lift machinery for the COMPONENT's custom-wit exports: those
    // live at the component level, not as core exports here. The worker can ONLY
    // call what its core module exports + what its own Linker imports provide.
    assert!(
        !exports.iter().any(|e| e == "cabi_realloc"),
        "worker core module has no canonical-ABI realloc -> cannot lift component types"
    );
    eprintln!(
        "STEP C: worker reaches plain imports (memory, record, thread-spawn) ONLY; \
         the component's lifted emit/read + any preview2 custom-wit imports require \
         the canonical ABI and the live component instance, which the worker never touches."
    );
}
