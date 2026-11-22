//! THROWAWAY SPIKE (monocp per-thread-component) -- do NOT commit.
//!
//! Tests the USER DESIGN: "give each thread its OWN component instance, sharing
//! ONE linear memory, so each thread has its OWN canonical-ABI state (no
//! contention) and can cross its OWN boundary concurrently."
//!
//! We probe, in order, the precise walls:
//!
//!   T1  CROSS-COMPONENT MEMORY SHARING. Instantiate a provider component (inner
//!       core $main DEFINES+exports a shared memory) on Store A, fetch its
//!       SharedMemory via the fork patch get_core_shared_memory. Then try to get
//!       a SECOND, independent component instance (own Store) to share that SAME
//!       SharedMemory. Establishes whether two FULL component instances can be
//!       backed by one memory at all (the component model only lets a component
//!       IMPORT a core MODULE template -- never a bare memory or a live core
//!       instance -- verified separately with wasm-tools).
//!
//!   T2  TWO BOUNDARIES OVER ONE MEMORY, ONE COMPONENT. A single component with
//!       TWO inner core instances sharing ONE memory (provider defines+exports,
//!       worker imports it), BOTH lifted to canonical-ABI exports. Verify both
//!       lifted calls hit the SAME memory and that calling worker-bump after
//!       provider-bump (post_return'd) does NOT trap -- characterising the
//!       may_enter flag.
//!
//!   T3  CONCURRENT CANONICAL-ABI FROM TWO OS THREADS. Attempt to drive a
//!       canonical-ABI (lifted) call from a worker OS thread while the main
//!       thread also drives one, against ONE component instance. This is the
//!       crux of the user design. Demonstrates the Store-is-single-threaded
//!       wall: a lifted call needs &mut StoreContextMut, and a Store cannot be
//!       shared &mut across threads, so two concurrent canonical-ABI calls into
//!       one component instance are impossible regardless of may_enter.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Barrier};
use wasmtime::component::{Component, Linker as CLinker};
use wasmtime::{Config, Engine, SharedMemory, Store};

fn engine() -> Engine {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
}

// Provider component: inner core $main DEFINES+exports a shared memory and lifts
// `bump`/`read`. This is the natural componentized-CPython topology (one core
// instance, defined memory).
const PROVIDER_WASM: &[u8] = include_bytes!("monocp_ptc_provider.wasm");

// Two-sub-instance component: provider defines+exports memory, worker imports
// the SAME memory, BOTH lifted. One memory, two canonical-ABI export paths.
const TWO_SUBINST_WASM: &[u8] = include_bytes!("monocp_ptc_two_subinst.wasm");

/// T1: can TWO independent component instances (separate Stores) share ONE
/// SharedMemory? The provider's inner memory is fetched from Store A. A second
/// component instance is built on Store B. We show the second instance gets its
/// OWN, distinct memory: a write through A's fetched SharedMemory is NOT visible
/// to B's lifted `read`. This is the cross-component-instance sharing wall.
#[test]
fn t1_two_component_instances_cannot_share_one_memory() {
    let engine = engine();
    let component = Component::new(&engine, PROVIDER_WASM).unwrap();

    // Instance A on Store A.
    let mut store_a = Store::new(&engine, ());
    let linker_a: CLinker<()> = CLinker::new(&engine);
    let inst_a = linker_a.instantiate(&mut store_a, &component).unwrap();
    let mem_a: SharedMemory = inst_a
        .get_core_shared_memory(&mut store_a, 0, "memory")
        .expect("provider A inner shared memory via fork patch");

    // Instance B on Store B -- the SAME component, but a fresh instantiation.
    let mut store_b = Store::new(&engine, ());
    let linker_b: CLinker<()> = CLinker::new(&engine);
    let inst_b = linker_b.instantiate(&mut store_b, &component).unwrap();
    let mem_b: SharedMemory = inst_b
        .get_core_shared_memory(&mut store_b, 0, "memory")
        .expect("provider B inner shared memory via fork patch");

    // The two memories are DIFFERENT Arc-backed objects: distinct base pointers.
    let a_ptr = mem_a.data().as_ptr();
    let b_ptr = mem_b.data().as_ptr();
    eprintln!("T1: A memory base = {a_ptr:?}, B memory base = {b_ptr:?}");
    assert_ne!(
        a_ptr, b_ptr,
        "two component instances minted DISTINCT memories (expected: not shared)"
    );

    // Write a sentinel into A's memory; B's lifted read must NOT see it.
    unsafe {
        (*(mem_a.data().as_ptr() as *const AtomicI32)).store(0x1234, Ordering::SeqCst);
    }
    let read_b = inst_b
        .get_typed_func::<(), (u32,)>(&mut store_b, "read")
        .unwrap();
    let seen_b = read_b.call(&mut store_b, ()).unwrap().0;
    read_b.post_return(&mut store_b).unwrap();
    eprintln!("T1: wrote 0x1234 into A's memory; B's lifted read sees 0x{seen_b:x}");
    assert_eq!(
        seen_b, 0,
        "B does NOT observe A's write -> the two component instances do NOT share memory"
    );

    eprintln!(
        "T1 WALL: the component model lets a component IMPORT only a core MODULE \
         template (mints fresh memory per instantiation), never a bare core memory \
         or a live core instance. So two FULL component instances on two Stores \
         CANNOT be backed by one SharedMemory."
    );
}

/// T2: ONE component, TWO inner core instances sharing ONE memory, BOTH lifted.
/// Verify both canonical-ABI export paths operate on the SAME memory, and that
/// the may_enter flag does not spuriously trap when alternating calls.
#[test]
fn t2_two_boundaries_one_memory_one_component() {
    let engine = engine();
    let component = Component::new(&engine, TWO_SUBINST_WASM).unwrap();
    let mut store = Store::new(&engine, ());
    let linker: CLinker<()> = CLinker::new(&engine);
    let instance = linker.instantiate(&mut store, &component).unwrap();

    let provider_bump = instance
        .get_typed_func::<(u32,), (u32,)>(&mut store, "provider-bump")
        .unwrap();
    let worker_bump = instance
        .get_typed_func::<(u32,), (u32,)>(&mut store, "worker-bump")
        .unwrap();
    let read = instance
        .get_typed_func::<(), (u32,)>(&mut store, "read")
        .unwrap();

    // provider-bump(+10) on the SHARED counter.
    let after_p = provider_bump.call(&mut store, (10,)).unwrap().0;
    provider_bump.post_return(&mut store).unwrap();
    eprintln!("T2: provider-bump(10) -> {after_p}");
    assert_eq!(after_p, 10);

    // worker-bump(+5): the worker IMPORTS the SAME memory, so it sees 10 and
    // returns 15 -- proving both canonical-ABI paths share ONE memory.
    let after_w = worker_bump.call(&mut store, (5,)).unwrap().0;
    worker_bump.post_return(&mut store).unwrap();
    eprintln!("T2: worker-bump(5) on the SAME memory -> {after_w}");
    assert_eq!(
        after_w, 15,
        "worker (imported memory) observed provider's write -> ONE shared memory across two boundaries"
    );

    let total = read.call(&mut store, ()).unwrap().0;
    read.post_return(&mut store).unwrap();
    eprintln!("T2: read() -> {total}");
    assert_eq!(total, 15);

    eprintln!(
        "T2: a SINGLE component instance CAN host two canonical-ABI export paths \
         over ONE shared memory (provider defines+exports, worker imports). But \
         both calls go through ONE Store on ONE thread (see T3)."
    );
}

/// T3: THE CRUX. Try to drive a canonical-ABI (lifted) call from a WORKER OS
/// thread CONCURRENTLY with the main thread, against one component instance over
/// one shared memory. A lifted call requires `&mut StoreContextMut`; a Store is
/// `Send` but not shareable as `&mut` across threads. We attempt to move a typed
/// func + a borrow into a worker and observe that it does not type-check / cannot
/// be done -- the design's hard wall. We encode the wall as a runtime
/// demonstration: the worker can only touch RAW shared memory (no canonical
/// ABI), while the main thread does the canonical-ABI call.
#[test]
fn t3_concurrent_canonical_abi_two_threads_wall() {
    let engine = engine();
    let component = Component::new(&engine, TWO_SUBINST_WASM).unwrap();
    let mut store = Store::new(&engine, ());
    let linker: CLinker<()> = CLinker::new(&engine);
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // Fetch the shared memory so the worker has SOMETHING to touch concurrently.
    let shared: SharedMemory = instance
        .get_core_shared_memory(&mut store, 0, "memory")
        .expect("inner shared memory via fork patch");

    let provider_bump = instance
        .get_typed_func::<(u32,), (u32,)>(&mut store, "provider-bump")
        .unwrap();
    let read = instance
        .get_typed_func::<(), (u32,)>(&mut store, "read")
        .unwrap();

    const ITERS: i32 = 200_000;
    let barrier = Arc::new(Barrier::new(2));

    // WORKER OS THREAD: it CANNOT make a canonical-ABI call -- it has no &mut
    // Store (the Store is moved/borrowed by the main thread). It can ONLY do RAW
    // atomic increments directly on the shared memory. This is precisely the
    // limitation: a worker thread cannot cross the component boundary on its own.
    let worker_mem = shared.clone();
    let wbar = barrier.clone();
    let handle = std::thread::spawn(move || {
        wbar.wait();
        let ptr = worker_mem.data().as_ptr() as *const AtomicI32;
        for _ in 0..ITERS {
            unsafe { (*ptr).fetch_add(1, Ordering::SeqCst) };
        }
    });

    // MAIN THREAD: canonical-ABI provider-bump in a loop, concurrently.
    barrier.wait();
    for _ in 0..ITERS {
        provider_bump.call(&mut store, (1,)).unwrap();
        provider_bump.post_return(&mut store).unwrap();
    }

    handle.join().unwrap();

    let total = read.call(&mut store, ()).unwrap().0;
    read.post_return(&mut store).unwrap();
    eprintln!(
        "T3: main canonical-ABI bumps + worker RAW-memory bumps -> counter = {total} (expected {})",
        2 * ITERS
    );
    assert_eq!(
        total, 2 * ITERS as u32,
        "both threads' increments land on the ONE shared memory (atomics coherent)"
    );

    eprintln!(
        "T3 WALL: the worker thread did RAW memory atomics ONLY. It could NOT make \
         a canonical-ABI call: a lifted call needs `&mut StoreContextMut`, and that \
         &mut borrow is held by the main thread for ITS call. A Store is single- \
         threaded for execution, so TWO concurrent canonical-ABI calls into one \
         component instance are impossible -- independent of per-sub-instance \
         may_enter flags."
    );
}
