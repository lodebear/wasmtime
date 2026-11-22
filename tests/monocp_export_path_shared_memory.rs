//! THROWAWAY EXPERIMENT (monocp) -- do NOT commit.
//!
//! FAITHFUL EXPORT-path test for the wasmtime fork patch.
//!
//! Goal: prove the host can fetch the *inner core* shared linear memory of a
//! COMPONENT whose inner core module DEFINES and EXPORTS a shared memory (this
//! mirrors the topology of our componentized CPython `$main` core instance),
//! and that the resulting `SharedMemory` handle is the SAME Arc-backed memory
//! the guest sees -- so a second OS thread can write a sentinel into it and the
//! guest observes the write via an atomic load.
//!
//! This is the path the prior spike (stage B) declared "unreachable" because no
//! host API existed. This test drives the new `Instance::get_core_shared_memory`
//! host method into existence (TDD: written to FAIL first, then implemented).

use std::sync::atomic::{AtomicI32, Ordering};
use wasmtime::component::{Component, Linker as CLinker};
use wasmtime::{Config, Engine, SharedMemory, Store};

fn engine() -> Engine {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
}

/// A component whose single inner core module ($main) DEFINES + EXPORTS a
/// shared memory plus the read / wasi_thread_start entry points. This matches
/// the real CPython componentized topology: one core instance, shared memory
/// exported by name "memory".
const COMPONENT_EXPORT_WAT: &str = r#"
(component
  (core module $main
    (memory (export "memory") 1 10 shared)
    (func (export "read") (result i32) (i32.atomic.load (i32.const 0)))
    (func (export "write") (param i32) (i32.atomic.store (i32.const 0) (local.get 0)))
    (func (export "wasi_thread_start") (param i32 i32)
      (i32.atomic.store (i32.const 0) (local.get 1)))
  )
  (core instance $i (instantiate $main))
  (core func $read (alias core export $i "read"))
  (func (export "read") (result u32) (canon lift (core func $read)))
)
"#;

#[test]
fn export_path_fetch_inner_shared_memory_cross_thread() {
    let engine = engine();
    let component = Component::new(&engine, COMPONENT_EXPORT_WAT)
        .expect("export-path component should compile");
    let mut store = Store::new(&engine, ());
    let linker: CLinker<()> = CLinker::new(&engine);
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // THE NEW HOST API UNDER TEST: fetch the inner core instance's shared
    // memory by (runtime core instance index, export name). For the CPython
    // topology there is exactly one inner core instance -> index 0.
    let shared: SharedMemory = instance
        .get_core_shared_memory(&mut store, 0, "memory")
        .expect("inner core shared memory 'memory' should be reachable from host");

    // Negative cases: wrong export name and out-of-range core instance index
    // must both return None (not panic, not a bogus handle).
    assert!(
        instance
            .get_core_shared_memory(&mut store, 0, "nope")
            .is_none(),
        "unknown export name must yield None"
    );
    assert!(
        instance
            .get_core_shared_memory(&mut store, 99, "memory")
            .is_none(),
        "out-of-range core instance index must yield None"
    );

    // The guest's lifted `read` function reads the first i32 atomically.
    let read = instance
        .get_typed_func::<(), (u32,)>(&mut store, "read")
        .unwrap();
    assert_eq!(read.call(&mut store, ()).unwrap().0, 0);
    read.post_return(&mut store).unwrap();

    // Spawn a second OS thread that writes a sentinel directly through the
    // SAME Arc-backed SharedMemory the guest is using. If get_core_shared_memory
    // handed back the real inner memory (not a copy), the guest will see it.
    let writer = shared.clone();
    let handle = std::thread::spawn(move || {
        let data = writer.data();
        let ptr = data.as_ptr() as *const AtomicI32;
        unsafe { (*ptr).store(0x4242, Ordering::SeqCst) };
    });
    handle.join().unwrap();

    // Guest observes the cross-thread write -> proves identity of the memory.
    assert_eq!(read.call(&mut store, ()).unwrap().0, 0x4242);
}
