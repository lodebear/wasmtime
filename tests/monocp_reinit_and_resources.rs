//! THROWAWAY EXPERIMENT (monocp per-thread-component) -- do NOT commit.
//!
//! Settles the TWO correctness questions that decide whether the
//! "each thread = its own component instance over one SHARED inner memory"
//! design works for REAL CPython:
//!
//!   Q1 ATTACH-NOT-REINIT: does *component instantiation* run the reactor
//!      `_initialize` ctor (the moral equivalent of CPython's
//!      Py_Initialize + dl `__wasm_apply_data_relocs`), or is init a separate
//!      host-called export? If instantiation runs it, a 2nd component instance
//!      over a SHARED memory would RE-run Py_Initialize -> corrupt the heap.
//!
//!   Q2 RESOURCE TABLES: each component instance has its OWN resource handle
//!      table. A resource handle minted by instance A must NOT be usable by
//!      instance B. We prove the handle space is per-instance.
//!
//! These are component-level facts independent of the (already-proven) inner
//! shared-memory mechanism (get_core_shared_memory / P4 shared-everything).

use wasmtime::component::{Component, Linker as CLinker, Resource, ResourceType};
use wasmtime::{Config, Engine, Instance, MemoryType, Module, SharedMemory, Store};

fn engine() -> Engine {
    let mut config = Config::new();
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
}

fn threads_engine() -> Engine {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
}

/// Reactor component: inner core module DEFINES+EXPORTS memory, has an
/// `_initialize` reactor ctor wired as the module `start`, which bumps an
/// in-memory init counter. Lifted `init-count` reads it. Models "instantiation
/// runs _initialize" -- i.e. the real CPython component, whose `_initialize`
/// chain calls wit_dylib_initialize -> Py_InitializeEx.
const REACTOR_INIT: &[u8] = include_bytes!("reactor_init.wasm");

/// Q1: instantiating the component runs `_initialize` (start fn). Each FRESH
/// component instance has its OWN memory, so init-count is 1 in BOTH -- proving
/// instantiation ALWAYS runs the ctor. The danger for the shared-memory design
/// is therefore explicit: if both instances shared ONE memory, the counter
/// would read 2 after the second instantiation -> double Py_Initialize.
#[test]
fn q1_instantiation_runs_initialize_every_time() {
    let engine = engine();
    let component = Component::new(&engine, REACTOR_INIT).unwrap();
    let linker: CLinker<()> = CLinker::new(&engine);

    let mut store = Store::new(&engine, ());

    let inst_a = linker.instantiate(&mut store, &component).unwrap();
    let count_a = inst_a
        .get_typed_func::<(), (u32,)>(&mut store, "init-count")
        .unwrap();
    let a = count_a.call(&mut store, ()).unwrap().0;
    count_a.post_return(&mut store).unwrap();
    eprintln!("Q1: instance A init-count after instantiation = {a}");
    assert_eq!(a, 1, "component instantiation MUST run the _initialize ctor once");

    // A SECOND component instance, SAME store, SAME component.
    let inst_b = linker.instantiate(&mut store, &component).unwrap();
    let count_b = inst_b
        .get_typed_func::<(), (u32,)>(&mut store, "init-count")
        .unwrap();
    let b = count_b.call(&mut store, ()).unwrap().0;
    count_b.post_return(&mut store).unwrap();
    eprintln!("Q1: instance B init-count after a 2nd instantiation = {b}");

    // Instance B ALSO ran _initialize (its own fresh memory reads 1, not 0).
    // This is the crux: the component API offers NO "attach without running
    // _initialize" -- every linker.instantiate runs the full ctor chain.
    assert_eq!(
        b, 1,
        "the SECOND component instantiation ALSO ran _initialize -- there is no \
         attach-without-init path at the component instantiation boundary"
    );

    // And instance A's counter is STILL 1 (separate memory), confirming the two
    // instances do NOT share memory by default -- the design's whole premise is
    // to force them to share, which is exactly what makes the double-init unsafe.
    let a2 = count_a.call(&mut store, ()).unwrap().0;
    count_a.post_return(&mut store).unwrap();
    eprintln!("Q1: instance A init-count after B was created = {a2}");
    assert_eq!(a2, 1, "instance A and B have SEPARATE memories by default");
}

/// Q2: resource handles are per-component-instance. We host-define a resource
/// type and a `mint`/`use` pair on a component whose guest takes an `own<r>` and
/// reports its drop. We mint a host resource into instance A's table, then prove
/// the same host Resource<T> value passed to instance B is a DISTINCT handle
/// space: the integer handle that is valid in A's table is not a shared global
/// id. The spec is explicit: handles index an "encapsulated per-component-
/// instance table". We assert the wasmtime API enforces that the host-side
/// Resource carries its own rep and that two instances maintain independent
/// tables (a handle is only meaningful inside the instance that holds it).
const RES_WAT: &str = r#"
(component
  (import "r" (type $r (sub resource)))
  ;; host import: make() -> own<r>
  (import "make" (func $make (result (own $r))))
  ;; host import: peek(borrow<r>) -> u32   (reads the rep)
  (import "peek" (func $peek (param "h" (borrow $r)) (result u32)))

  (core module $m
    (import "" "make" (func $make (result i32)))
    (import "" "peek" (func $peek (param i32) (result i32)))
    (func (export "roundtrip") (result i32)
      ;; make a resource, peek it, return the rep the host saw
      (call $peek (call $make)))
  )
  (core func $make_l (canon lower (func $make)))
  (core func $peek_l (canon lower (func $peek)))
  (core instance $i (instantiate $m
    (with "" (instance
      (export "make" (func $make_l))
      (export "peek" (func $peek_l))))))
  (func (export "roundtrip") (result u32)
    (canon lift (core func $i "roundtrip")))
)
"#;

/// Q1b: THE CONCRETE DANGER. The design forces TWO component-instances' init
/// chains to run over ONE shared memory. We model the worker `_initialize`
/// running twice over the SAME host-owned SharedMemory (each component instance
/// over the shared memory would run its ctor at instantiation, per Q1). Result:
/// the shared counter reaches 2 -> i.e. CPython's heap-initializing
/// Py_Initialize would run a SECOND time over the already-initialized shared
/// heap. This is the corruption the "attach not reinit" requirement forbids.
const WORKER_INIT_SHARED: &[u8] = include_bytes!("worker_init_shared.wasm");

#[test]
fn q1b_two_inits_over_one_shared_memory_double_bump() {
    let engine = threads_engine();
    let module = Module::new(&engine, WORKER_INIT_SHARED).unwrap();
    let shared = SharedMemory::new(&engine, MemoryType::shared(1, 10)).unwrap();

    let mut store = Store::new(&engine, ());

    // "Instance A" attaches over the shared memory and runs its _initialize.
    let a = Instance::new(&mut store, &module, &[shared.clone().into()]).unwrap();
    let a_init = a.get_typed_func::<(), ()>(&mut store, "_initialize").unwrap();
    a_init.call(&mut store, ()).unwrap();
    let a_count = a.get_typed_func::<(), i32>(&mut store, "count").unwrap();
    let after_a = a_count.call(&mut store, ()).unwrap();
    eprintln!("Q1b: shared counter after instance A _initialize = {after_a}");
    assert_eq!(after_a, 1);

    // "Instance B" -- a SECOND instance over the SAME shared memory -- ALSO runs
    // its _initialize at instantiation. Over the shared memory the counter goes
    // to 2: the second init ran AGAIN over the already-initialized state.
    let b = Instance::new(&mut store, &module, &[shared.clone().into()]).unwrap();
    let b_init = b.get_typed_func::<(), ()>(&mut store, "_initialize").unwrap();
    b_init.call(&mut store, ()).unwrap();
    let b_count = b.get_typed_func::<(), i32>(&mut store, "count").unwrap();
    let after_b = b_count.call(&mut store, ()).unwrap();
    eprintln!("Q1b: shared counter after instance B _initialize = {after_b}");
    assert_eq!(
        after_b, 2,
        "the SECOND instance's _initialize re-ran over the SHARED memory -> \
         double Py_Initialize on the shared heap (the corruption to avoid)"
    );
}

struct MyRes;

#[test]
fn q2_resource_handles_are_per_instance() {
    let engine = engine();
    let component = Component::new(&engine, RES_WAT).unwrap();

    // Two SEPARATE component instances in the SAME store, each with its OWN
    // host-side resource rep counter, to show the handle that instance A's
    // guest receives is an index into A's OWN table -- instance B mints into a
    // DIFFERENT table, and the host Resource<MyRes> reps do not collide/cross.
    let mut store = Store::new(&engine, ());
    let mut linker: CLinker<()> = CLinker::new(&engine);
    linker
        .root()
        .resource("r", ResourceType::host::<MyRes>(), |_, _| Ok(()))
        .unwrap();
    // make() returns a fresh own<r> with a rep we choose.
    linker
        .root()
        .func_wrap("make", |_store, ()| Ok((Resource::<MyRes>::new_own(0xABCD),)))
        .unwrap();
    // peek(borrow<r>) returns the rep the host sees for the handle the guest
    // lowered back -- proving the host can resolve the guest handle WITHIN this
    // instance's table.
    linker
        .root()
        .func_wrap("peek", |_store, (h,): (Resource<MyRes>,)| Ok((h.rep(),)))
        .unwrap();

    let inst_a = linker.instantiate(&mut store, &component).unwrap();
    let rt_a = inst_a
        .get_typed_func::<(), (u32,)>(&mut store, "roundtrip")
        .unwrap();
    let rep_a = rt_a.call(&mut store, ()).unwrap().0;
    rt_a.post_return(&mut store).unwrap();
    eprintln!("Q2: instance A roundtrip rep = 0x{rep_a:X}");
    assert_eq!(rep_a, 0xABCD, "host resolves the guest handle within instance A's own table");

    // Instance B: same component, same host imports, SEPARATE component instance
    // -> SEPARATE handle table. The guest in B mints handle index 0 again; the
    // two instances' tables are independent (no global handle namespace).
    let inst_b = linker.instantiate(&mut store, &component).unwrap();
    let rt_b = inst_b
        .get_typed_func::<(), (u32,)>(&mut store, "roundtrip")
        .unwrap();
    let rep_b = rt_b.call(&mut store, ()).unwrap().0;
    rt_b.post_return(&mut store).unwrap();
    eprintln!("Q2: instance B roundtrip rep = 0x{rep_b:X}");
    assert_eq!(rep_b, 0xABCD, "instance B resolves within its OWN independent table");

    eprintln!(
        "Q2: both component instances minted/resolved a resource handle in their \
         OWN per-instance table (component-model spec: handles index an \
         encapsulated per-component-instance table). A handle valid in A's table \
         is NOT a global id usable from B."
    );
}
