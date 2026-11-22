//! THROWAWAY EXPERIMENT (monocp) -- do NOT commit.
//!
//! FULL thread-spawn-through-a-component flow for preview2 + STATIC + threads.
//!
//! Extends exp-4 (get_core_shared_memory proves memory IDENTITY across an OS
//! thread, writer joined-before-read) to the REAL wasi-threads mechanism:
//! re-instantiate the inner core module on a worker OS thread sharing the
//! component's inner core memory, run wasi_thread_start CONCURRENTLY with the
//! main thread, and verify atomics on a shared counter (no lost updates).
//!
//! We probe, in order, the precise walls:
//!   P0  defined-vs-imported shared memory semantics for re-instantiation
//!   P1  raw core re-instantiation cross-thread on a DEFINED shared memory
//!   P2  the same but on an IMPORTED shared memory (classic wasi-threads)
//!   P3  through a COMPONENT: fetch inner memory + try to re-instantiate inner

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use wasmtime::component::{Component, Linker as CLinker};
use wasmtime::{Config, Engine, Instance, MemoryType, Module, SharedMemory, Store};

fn engine() -> Engine {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
}

/// P0: Does wasmtime SHARE a DEFINED shared memory across two instances of the
/// SAME module in the SAME engine+store? wasi-libc static builds DEFINE their
/// shared memory; wasi-threads re-instantiates per worker. If defined memories
/// are per-instance (not deduped), re-instantiation gives each worker its OWN
/// memory -> threads cannot share state. This is the foundational question.
const DEFINE_MEM_WAT: &str = r#"
(module
  (memory (export "memory") 1 10 shared)
  (func (export "read") (result i32) (i32.atomic.load (i32.const 0)))
  (func (export "write") (param i32) (i32.atomic.store (i32.const 0) (local.get 0)))
)
"#;

#[test]
fn p0_defined_shared_memory_is_per_instance_not_shared() {
    let engine = engine();
    let module = Module::new(&engine, DEFINE_MEM_WAT).unwrap();

    let mut store = Store::new(&engine, ());
    let a = Instance::new(&mut store, &module, &[]).unwrap();
    let b = Instance::new(&mut store, &module, &[]).unwrap();

    let a_write = a.get_typed_func::<i32, ()>(&mut store, "write").unwrap();
    let b_read = b.get_typed_func::<(), i32>(&mut store, "read").unwrap();

    a_write.call(&mut store, 0x55).unwrap();
    let seen = b_read.call(&mut store, ()).unwrap();

    // If DEFINED shared memories were deduped/shared across instances, b would
    // see 0x55. They are NOT: each instance defines its own memory.
    eprintln!("P0: instance-B observed 0x{seen:x} after instance-A wrote 0x55");
    assert_eq!(
        seen, 0,
        "DEFINED shared memory is per-instance: re-instantiation does NOT share it"
    );
}

/// P2: classic wasi-threads -- IMPORTED shared memory, host-created, re-supplied
/// to a worker re-instantiation. Worker runs wasi_thread_start CONCURRENTLY with
/// main. This is the mechanism crates/wasi-threads/src/lib.rs implements. We
/// reproduce it WITHOUT a component to establish the working baseline + real
/// concurrency (no join-before-read).
const IMPORT_MEM_WAT: &str = r#"
(module
  (import "env" "memory" (memory 1 10 shared))
  ;; busy-loop N atomic increments at addr 0
  (func $bump (param $n i32)
    (local $i i32)
    (block $done (loop $loop
      (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
      (drop (i32.atomic.rmw.add (i32.const 0) (i32.const 1)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loop)))
  )
  (func (export "compute") (param $n i32) (call $bump (local.get $n)))
  (func (export "wasi_thread_start") (param $tid i32) (param $arg i32)
    (call $bump (local.get $arg)))
  (func (export "read") (result i32) (i32.atomic.load (i32.const 0)))
)
"#;

#[test]
fn p2_imported_memory_concurrent_worker_no_lost_updates() {
    let engine = engine();
    let module = Module::new(&engine, IMPORT_MEM_WAT).unwrap();
    let shared = SharedMemory::new(&engine, MemoryType::shared(1, 10)).unwrap();

    const ITERS: i32 = 200_000;

    // Worker re-instantiates the SAME module in its OWN store, sharing the
    // host-owned SharedMemory via the import -- exactly the wasi-threads model
    // (lib.rs:67-95 instantiate-per-thread; lib.rs:170-171 define imported mem).
    let worker_engine = engine.clone();
    let worker_module = module.clone();
    let worker_mem = shared.clone();
    let handle = std::thread::spawn(move || {
        let mut wstore = Store::new(&worker_engine, ());
        let winst =
            Instance::new(&mut wstore, &worker_module, &[worker_mem.into()]).unwrap();
        let start = winst
            .get_typed_func::<(i32, i32), ()>(&mut wstore, "wasi_thread_start")
            .unwrap();
        // wasi_thread_start(tid=1, arg=ITERS)
        start.call(&mut wstore, (1, ITERS)).unwrap();
    });

    // Main thread runs CONCURRENTLY (NOT joined first) on its own store/instance.
    let mut store = Store::new(&engine, ());
    let inst = Instance::new(&mut store, &module, &[shared.clone().into()]).unwrap();
    let compute = inst.get_typed_func::<i32, ()>(&mut store, "compute").unwrap();
    compute.call(&mut store, ITERS).unwrap();

    handle.join().unwrap();

    let read = inst.get_typed_func::<(), i32>(&mut store, "read").unwrap();
    let total = read.call(&mut store, ()).unwrap();
    eprintln!("P2: concurrent counter = {total} (expected {})", 2 * ITERS);
    assert_eq!(
        total,
        2 * ITERS,
        "atomic increments from BOTH threads must all land -> no lost updates"
    );
}

/// P3: through the COMPONENT. Fetch the inner core shared memory (exp-4 API),
/// then attempt to drive the full worker flow. The component's inner core
/// module DEFINES its memory (real wasi-sdk topology, confirmed). We probe what
/// the host can and cannot do to re-instantiate the inner core module on a
/// worker thread sharing that memory.
const THREADED_COMPONENT: &[u8] = include_bytes!("monocp_threaded_component.wasm");

#[test]
fn p3_component_inner_reinstantiation_wall() {
    let engine = engine();
    let component = Component::new(&engine, THREADED_COMPONENT).unwrap();
    let mut store = Store::new(&engine, ());
    let mut linker: CLinker<()> = CLinker::new(&engine);

    // Satisfy the component's `monocp:threads/threading.thread-spawn` import
    // with a host stub. (In a real impl this is where wasi-threads spawn lives.)
    linker
        .instance("monocp:threads/threading")
        .unwrap()
        .func_wrap(
            "thread-spawn",
            |_caller: wasmtime::StoreContextMut<'_, ()>, (arg,): (i32,)| Ok((arg,)),
        )
        .unwrap_or_else(|e| panic!("link thread-spawn: {e}"));

    let instance = linker.instantiate(&mut store, &component).unwrap();

    // Probe which runtime core-instance index holds the exported memory.
    for idx in 0..6u32 {
        let got = instance.get_core_shared_memory(&mut store, idx, "memory");
        eprintln!("P3 probe: get_core_shared_memory(idx={idx}, \"memory\") -> {}",
            if got.is_some() { "Some" } else { "None" });
    }
    // FINDING: the inner core $main is RUNTIME instance index 0, NOT the index 1
    // the wasm-tools text dump implies. The `canon lower` shim that satisfies the
    // thread-spawn import is NOT a real `InstantiateModule` initializer, so it
    // gets no RuntimeInstanceIndex -- only actual core-module instantiations do.
    let shared: SharedMemory = instance
        .get_core_shared_memory(&mut store, 0, "memory")
        .expect("inner core $main shared memory reachable via exp-4 patch");
    eprintln!(
        "P3: fetched inner core shared memory, {} bytes",
        shared.data().len()
    );

    // Prove memory IDENTITY (exp-4 level): main runs compute, worker writes a
    // sentinel directly into the SAME Arc-backed SharedMemory, joined first.
    let compute = instance
        .get_typed_func::<(i32,), (i32,)>(&mut store, "compute")
        .unwrap();
    let n = compute.call(&mut store, (100,)).unwrap().0;
    compute.post_return(&mut store).unwrap();
    eprintln!("P3: component compute(100) on main store -> counter = {n}");
    assert_eq!(n, 100);

    let writer = shared.clone();
    let h = std::thread::spawn(move || {
        let ptr = writer.data().as_ptr() as *const AtomicI32;
        unsafe { (*ptr).fetch_add(100, Ordering::SeqCst) };
    });
    h.join().unwrap();

    let read = instance
        .get_typed_func::<(), (i32,)>(&mut store, "read")
        .unwrap();
    let after = read.call(&mut store, ()).unwrap().0;
    read.post_return(&mut store).unwrap();
    eprintln!("P3: after cross-thread direct +100 -> counter = {after}");
    assert_eq!(after, 200, "cross-thread write visible: memory identity holds");

    // ATTEMPT THE FULL FLOW: fetch the inner core MODULE (added probe patch),
    // then on a worker thread RE-INSTANTIATE it on a fresh store and run
    // wasi_thread_start. Discover the concrete wall.
    let inner: Module = instance
        .get_core_instance_module(&mut store, 0)
        .expect("inner core module reachable via probe patch");
    eprintln!(
        "P3: fetched inner core MODULE; imports = {:?}",
        inner
            .imports()
            .map(|i| format!("{}::{}", i.module(), i.name()))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "P3: inner module memory imports? {}",
        inner.imports().any(|i| i.ty().memory().is_some())
    );

    // The inner module DEFINES its memory (component-model requirement: the main
    // module cannot import memory -- confirmed via wasm-tools, error: "main
    // module cannot import memory"). So Instance::new cannot be handed the SAME
    // SharedMemory; the worker re-instantiation will MINT A FRESH memory.
    // Prove it: counter was 200; a fresh worker instance sees 0.
    let worker_engine = engine.clone();
    let worker_module = inner.clone();
    let observed = std::thread::spawn(move || {
        let mut wstore = Store::new(&worker_engine, ());
        // The inner module imports `monocp:threads/threading::thread-spawn`. To
        // re-instantiate raw, the worker must satisfy that import itself with a
        // core func of matching type (i32)->i32. (Through the component this
        // import was a canon-lower shim wired into the component graph -- NOT
        // reconstructible from the host.)
        let stub = wasmtime::Func::wrap(&mut wstore, |a: i32| a);
        let winst = Instance::new(&mut wstore, &worker_module, &[stub.into()])
            .expect("worker re-instantiation of inner core module");
        let read = winst
            .get_typed_func::<(), i32>(&mut wstore, "read")
            .unwrap();
        read.call(&mut wstore, ()).unwrap()
    })
    .join()
    .unwrap();

    eprintln!(
        "P3 WALL: main-store counter = 200, but worker's FRESH re-instantiation \
         observes counter = {observed} (its own minted memory). DEFINED memory is \
         per-instance (P0); component model forbids imported memory in the main \
         module, so worker re-instantiation CANNOT share the inner memory."
    );
    assert_eq!(
        observed, 0,
        "worker re-instantiation minted a NEW memory -> no sharing (the wall)"
    );
    let _ = Arc::new(());
}

/// P4: THE BREAKTHROUGH PATH. A "shared-everything-linking" component:
///   - a PROVIDER inner core module DEFINES+EXPORTS the shared memory,
///   - a WORKER inner core module IMPORTS that shared memory (a NON-main module
///     CAN import memory -- only the *main* module is forbidden from doing so),
///   - thread-spawn is a component import lowered into the worker.
/// The host fetches the PROVIDER's shared memory via get_core_shared_memory and
/// the WORKER's core Module via get_core_instance_module, then on a real worker
/// OS thread RE-INSTANTIATES the worker module supplying that SAME shared memory
/// + a thread-spawn stub, and runs wasi_thread_start CONCURRENTLY with main.
/// Verifies atomics on the shared counter: REAL concurrency, no lost updates.
const SHARED_EVERYTHING: &[u8] = include_bytes!("monocp_shared_everything.wasm");

#[test]
fn p4_shared_everything_concurrent_worker_through_component() {
    let engine = engine();
    let component = Component::new(&engine, SHARED_EVERYTHING).unwrap();
    let mut store = Store::new(&engine, ());
    let mut linker: CLinker<()> = CLinker::new(&engine);
    linker
        .instance("monocp:threads/threading")
        .unwrap()
        .func_wrap(
            "thread-spawn",
            |_c: wasmtime::StoreContextMut<'_, ()>, (arg,): (i32,)| Ok((arg,)),
        )
        .unwrap();
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // Provider core instance = runtime index 0 (first real InstantiateModule);
    // worker core instance = runtime index 1. Probe to confirm.
    for idx in 0..4u32 {
        eprintln!(
            "P4 probe: shared mem at core idx {idx} = {}",
            instance
                .get_core_shared_memory(&mut store, idx, "memory")
                .is_some()
        );
    }
    let shared: SharedMemory = instance
        .get_core_shared_memory(&mut store, 0, "memory")
        .expect("provider core shared memory");
    let worker_mod: Module = instance
        .get_core_instance_module(&mut store, 1)
        .expect("worker core module");
    eprintln!(
        "P4: worker module imports = {:?}",
        worker_mod
            .imports()
            .map(|i| format!("{}::{}", i.module(), i.name()))
            .collect::<Vec<_>>()
    );
    let worker_imports_mem = worker_mod.imports().any(|i| i.ty().memory().is_some());
    eprintln!("P4: worker module IMPORTS its memory? {worker_imports_mem}");
    assert!(
        worker_imports_mem,
        "worker (non-main) module must import its shared memory"
    );

    const ITERS: i32 = 200_000;

    // WORKER OS THREAD: re-instantiate the worker core module on a fresh store,
    // sharing the PROVIDER's SharedMemory via the imported `env::memory`, plus a
    // thread-spawn stub for `wasi::thread-spawn`. Run wasi_thread_start.
    let weng = engine.clone();
    let wmod = worker_mod.clone();
    let wmem = shared.clone();
    let handle = std::thread::spawn(move || {
        let mut wstore = Store::new(&weng, ());
        // Build imports in the order the worker module declares them.
        let mut externs: Vec<wasmtime::Extern> = Vec::new();
        for imp in wmod.imports() {
            if imp.ty().memory().is_some() {
                externs.push(wmem.clone().into());
            } else {
                // wasi::thread-spawn (i32)->i32 stub
                let f = wasmtime::Func::wrap(&mut wstore, |a: i32| a);
                externs.push(f.into());
            }
        }
        let winst = Instance::new(&mut wstore, &wmod, &externs)
            .expect("worker re-instantiation sharing provider memory");
        let start = winst
            .get_typed_func::<(i32, i32), ()>(&mut wstore, "wasi_thread_start")
            .unwrap();
        start.call(&mut wstore, (1, ITERS)).unwrap();
    });

    // MAIN THREAD runs CONCURRENTLY through the COMPONENT's lifted `compute`.
    let compute = instance
        .get_typed_func::<(i32,), (i32,)>(&mut store, "compute")
        .unwrap();
    compute.call(&mut store, (ITERS,)).unwrap();
    compute.post_return(&mut store).unwrap();

    handle.join().unwrap();

    let read = instance
        .get_typed_func::<(), (i32,)>(&mut store, "read")
        .unwrap();
    let total = read.call(&mut store, ()).unwrap().0;
    read.post_return(&mut store).unwrap();
    eprintln!("P4: concurrent counter through component = {total} (expected {})", 2 * ITERS);
    assert_eq!(
        total,
        2 * ITERS,
        "main (via component) + worker (re-instantiated) increments all land: \
         FULL preview2+static+threads flow works with shared-everything linking"
    );
}
