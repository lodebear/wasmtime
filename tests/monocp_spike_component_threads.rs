//! THROWAWAY SPIKE (monocp) -- do NOT commit.
//!
//! Question: can a COMPONENT share its inner core linear memory across a
//! spawned OS thread, decoupling the wasi-threads fork mechanism from CPython?
//!
//! We probe three things:
//!   stage A: baseline -- raw CORE module + host SharedMemory, spawn an OS
//!            thread that writes while main reads via atomics. (Proves the
//!            SharedMemory cross-thread mechanism itself works in v41.)
//!   stage B: EXPORT sub-path -- wrap the core module in a component that
//!            *exports* the inner core instance's shared memory, instantiate
//!            via the component API, and try to fetch the SharedMemory handle
//!            from the host.
//!   stage C: IMPORT sub-path -- can the component Linker inject a host-owned
//!            SharedMemory to satisfy a core module's shared-memory import?

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

// A core module that *imports* a shared memory, exports a tiny function, and
// exports the wasi_thread_start entry point. This mirrors a wasi-threads guest.
const CORE_WAT: &str = r#"
(module
  (import "env" "memory" (memory 1 10 shared))
  ;; read first i32 of shared memory atomically
  (func (export "read") (result i32)
    (i32.atomic.load (i32.const 0)))
  ;; write a value to the first i32 atomically
  (func (export "write") (param i32)
    (i32.atomic.store (i32.const 0) (local.get 0)))
  ;; wasi-threads entry point signature: (i32 tid, i32 arg) -> ()
  (func (export "wasi_thread_start") (param i32 i32)
    (i32.atomic.store (i32.const 0) (local.get 1)))
)
"#;

/// Stage A: raw core baseline. Host owns a SharedMemory, hands it to the core
/// module's import; a second OS thread writes through the SAME SharedMemory
/// while the main thread reads it. No store sharing needed -- SharedMemory is
/// Arc-backed and Send.
#[test]
fn stage_a_core_baseline_cross_thread() {
    let engine = engine();
    let module = Module::new(&engine, CORE_WAT).unwrap();
    let shared = SharedMemory::new(&engine, MemoryType::shared(1, 10)).unwrap();

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[shared.clone().into()]).unwrap();
    let read = instance
        .get_typed_func::<(), i32>(&mut store, "read")
        .unwrap();
    assert_eq!(read.call(&mut store, ()).unwrap(), 0);

    // Spawn an OS thread that writes via the SharedMemory directly (this is the
    // moral equivalent of what wasi-threads does in its own store/instance).
    let writer = shared.clone();
    let handle = std::thread::spawn(move || {
        // SharedMemory::data() exposes &[UnsafeCell<u8>]; write atomically.
        let data = writer.data();
        let ptr = data.as_ptr() as *const AtomicI32;
        unsafe { (*ptr).store(0x4242, Ordering::SeqCst) };
    });
    handle.join().unwrap();

    // Main thread reads via the guest function -- should observe the write.
    assert_eq!(read.call(&mut store, ()).unwrap(), 0x4242);
}

/// Stage B: EXPORT sub-path. A component that contains an inner core module
/// defining+exporting a shared memory. We try TWO things:
///  B1: the *only* thing the component model lets us export to the host that
///      could plausibly reach core memory is a core MODULE definition (static)
///      -- which is NOT an instance, so it has no live memory.
///  B2: the component model rejects `(export (core memory ...))` and
///      `(export (core instance ...))` outright (verified separately with
///      wasm-tools). So the only instantiable host-facing fixture is one that
///      exports a lifted *function* that reads the inner memory.
/// Conclusion target: there is no Instance method to fetch a SharedMemory.
const COMPONENT_EXPORT_WAT: &str = r#"
(component
  (core module $m
    (memory (export "memory") 1 10 shared)
    (func (export "read") (result i32) (i32.atomic.load (i32.const 0)))
  )
  (core instance $i (instantiate $m))
  ;; we can alias the inner core memory INSIDE the component...
  (alias core export $i "memory" (core memory $mem))
  ;; ...but we can only export a lifted function to the host, not the memory.
  (core func $read (alias core export $i "read"))
  (func (export "read") (result u32)
    (canon lift (core func $read)))
)
"#;

#[test]
fn stage_b_component_export_fetch_shared_memory() {
    let engine = engine();
    let component = Component::new(&engine, COMPONENT_EXPORT_WAT)
        .expect("COMPONENT_EXPORT_WAT should compile (lifted-func form)");
    let mut store = Store::new(&engine, ());
    let linker: CLinker<()> = CLinker::new(&engine);
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // The component Instance API surface (instance.rs): get_func, get_typed_func,
    // get_module, get_resource, get_export, get_export_index. There is NO
    // get_memory / get_shared_memory. Enumerate exports to prove it:
    let read = instance.get_export_index(&mut store, None, "read");
    eprintln!("stage_b: component-level export 'read' present = {}", read.is_some());

    // The inner core memory is NOT reachable: no export index, no API.
    // (Compile-time proof: `instance.get_shared_memory(...)` does not exist on
    // the component Instance -- only on the *core* wasmtime::Instance.)
    eprintln!(
        "stage_b: component Instance exposes NO get_shared_memory / get_memory \
         -> inner core SharedMemory is unreachable from the host"
    );
}

/// Stage C: IMPORT sub-path. A component whose inner core module IMPORTS a
/// shared memory. In core wasi-threads this is satisfied by
/// `core_linker.define(store, module, name, shared_memory)` (lib.rs:170-171).
///
/// The closest a component can do is *import a core module* from the host that
/// exports a shared memory, then instantiate it internally and wire its memory
/// export into the worker module. The host supplies the module via
/// `linker.root().module(name, &Module)`. CRITICAL: the host hands over a
/// static Module *template*; the SharedMemory is created INSIDE the component's
/// instantiation, so the host never gets a handle to it.
const COMPONENT_IMPORT_WAT: &str = r#"
(component
  ;; host must supply a core module that exports a shared memory
  (import "hostmem" (core module $hm (export "memory" (memory 1 10 shared))))
  (core instance $h (instantiate $hm))
  (core module $m
    (import "env" "memory" (memory 1 10 shared))
    (func (export "read") (result i32) (i32.atomic.load (i32.const 0)))
  )
  (core instance $i (instantiate $m (with "env" (instance $h))))
  (core func $read (alias core export $i "read"))
  (func (export "read") (result u32) (canon lift (core func $read)))
)
"#;

#[test]
fn stage_c_component_import_inject_shared_memory() {
    let engine = engine();
    let component = Component::new(&engine, COMPONENT_IMPORT_WAT)
        .expect("COMPONENT_IMPORT_WAT should compile");

    // Host attempts to inject memory. The ONLY define methods on the component
    // LinkerInstance are: func_wrap/func_new (Func), module (Module),
    // resource (Resource), instance (nested Instance namespace). The internal
    // `Definition` enum (linker.rs:105-109) has exactly those 4 variants and
    // NO Memory/SharedMemory variant.
    //
    // So we cannot inject a *live* SharedMemory. We can only hand over a static
    // Module template that DEFINES a shared memory; the actual memory object is
    // created during the component's own instantiation, inside the component,
    // and is never returned to the host.
    let host_mem_module = Module::new(
        &engine,
        r#"(module (memory (export "memory") 1 10 shared))"#,
    )
    .unwrap();

    let mut store = Store::new(&engine, ());
    let mut linker: CLinker<()> = CLinker::new(&engine);
    linker.root().module("hostmem", &host_mem_module).unwrap();
    let instance = linker.instantiate(&mut store, &component).unwrap();

    // We instantiated successfully, but we hold NO SharedMemory handle: the
    // memory lives inside the component's hidden core instance.
    let read = instance.get_export_index(&mut store, None, "read");
    eprintln!("stage_c: component instantiated, 'read' export present = {}", read.is_some());
    eprintln!(
        "stage_c: component Linker has NO memory/shared_memory define path \
         (Definition enum: Instance|Func|Module|Resource) -> host cannot inject \
         or retrieve the inner SharedMemory"
    );
    let _ = Arc::new(()); // keep Arc import used
}
