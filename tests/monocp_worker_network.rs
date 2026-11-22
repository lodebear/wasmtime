//! THROWAWAY EXPERIMENT (monocp p2-threads REAL-NETWORK) -- do NOT commit.
//!
//! Network analog of the proven worker-real-WASI-file spike
//! (monocp_worker_real_wasi.rs). That spike proved a worker OS thread can run
//! REAL WASI preview1 filesystem syscalls on its OWN Store + OWN plain Linker,
//! concurrently with the main thread driving a preview2 component (canonical
//! ABI), without touching the component canonical-ABI state.
//!
//! GOAL proven here: a worker OS thread, on its OWN `Store` + OWN plain
//! `wasmtime::Linker`, wired with PLAIN host networking imports (modelled on
//! monocp's real plain core imports: `monocp.tcp_connect/tcp_write/tcp_read/
//! tcp_close`) backed by a SHARED, Mutex-protected host connection registry that
//! does REAL `std::net::TcpStream` socket I/O, performs an ACTUAL TCP round-trip
//! to a real localhost echo server -- CONCURRENTLY with the main thread driving
//! a preview2 component's lifted `emit`/`read` (canonical ABI) AND the main
//! thread doing its OWN real tcp round-trip through the SAME shared registry.
//! Then VERIFY:
//!   (a) the worker's round-tripped echo bytes match the sent payload exactly
//!       (read back out of the worker module's OWN linear memory by the host),
//!   (b) the shared host connection registry coherently saw BOTH threads'
//!       connections (>=2 connections, both threads' payloads echoed),
//!   (c) no trap (worker_entry returned a non-negative byte count),
//!   (d) the component's canonical-ABI state was never touched: the worker used
//!       ONLY its own plain Linker; the main component instance remains callable
//!       and consistent afterwards (a fresh lifted call still works).
//!
//! The worker core module (tests/monocp_worker_net.wasm) is a FREESTANDING
//! wasm32 program built with wasi-sdk-27 clang that imports plain host fns
//! `monocp.tcp_connect/tcp_write/tcp_read/tcp_close` directly and exports
//! `worker_entry`. It owns its single linear memory (exported "memory").

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use wasmtime::component::{Component, Linker as CLinker};
use wasmtime::{Caller, Config, Engine, Extern, Linker, Module, Store};

/// Worker core module: plain `monocp.tcp_*` imports + `worker_entry` export.
const WORKER_WASM: &[u8] = include_bytes!("monocp_worker_net.wasm");

/// Reuse the prior spike's STEP_B component for the MAIN thread's concurrent
/// component canonical-ABI work (lifted `emit`/`read`).
const STEP_B: &[u8] = include_bytes!("stepB.wasm");

fn engine() -> Engine {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
}

// ---------------------------------------------------------------------------
// SHARED, thread-safe host network state. This mirrors how the monocp preview1
// host shares its fdTable/networks across threads (state behind a Mutex,
// instantiate-per-thread). BOTH the worker store's plain imports and the main
// thread's own tcp helper hit this SAME registry.
// ---------------------------------------------------------------------------
#[derive(Default)]
struct NetRegistry {
    next_fd: i32,
    conns: HashMap<i32, TcpStream>,
    /// observability: which (thread-label, peer-addr) pairs connected.
    connect_log: Vec<(String, String)>,
}

#[derive(Clone, Default)]
struct SharedNet(Arc<Mutex<NetRegistry>>);

impl SharedNet {
    /// REAL connect via std::net. Records the connection in the shared registry.
    fn connect(&self, label: &str, host: &str, port: u16) -> i32 {
        match TcpStream::connect((host, port)) {
            Ok(stream) => {
                stream.set_nodelay(true).ok();
                let peer = stream
                    .peer_addr()
                    .map(|a| a.to_string())
                    .unwrap_or_default();
                let mut reg = self.0.lock().unwrap();
                let fd = reg.next_fd;
                reg.next_fd += 1;
                reg.conns.insert(fd, stream);
                reg.connect_log.push((label.to_string(), peer));
                fd
            }
            Err(_) => -1, // negative errno (simplified)
        }
    }
    fn write(&self, fd: i32, buf: &[u8]) -> i32 {
        let mut reg = self.0.lock().unwrap();
        match reg.conns.get_mut(&fd) {
            Some(s) => match s.write(buf) {
                Ok(n) => n as i32,
                Err(_) => -5,
            },
            None => -9, // EBADF-ish
        }
    }
    fn read(&self, fd: i32, buf: &mut [u8]) -> i32 {
        let mut reg = self.0.lock().unwrap();
        match reg.conns.get_mut(&fd) {
            Some(s) => match s.read(buf) {
                Ok(n) => n as i32,
                Err(_) => -5,
            },
            None => -9,
        }
    }
    fn close(&self, fd: i32) -> i32 {
        let mut reg = self.0.lock().unwrap();
        match reg.conns.remove(&fd) {
            Some(_) => 0,
            None => -9,
        }
    }
}

/// Per-store state for the WORKER store. Carries the shared registry + a thread
/// label so the registry can attribute connections.
struct WorkerState {
    net: SharedNet,
    label: String,
}

/// Helper: read `len` bytes at `ptr` out of the worker module's OWN "memory".
fn read_guest(caller: &mut Caller<'_, WorkerState>, ptr: i32, len: i32) -> Vec<u8> {
    let mem = match caller.get_export("memory") {
        Some(Extern::Memory(m)) => m,
        _ => panic!("worker module must export 'memory'"),
    };
    let data = mem.data(&mut *caller);
    let start = ptr as usize;
    data[start..start + len as usize].to_vec()
}

/// Helper: write `bytes` at `ptr` into the worker module's OWN "memory".
fn write_guest(caller: &mut Caller<'_, WorkerState>, ptr: i32, bytes: &[u8]) {
    let mem = match caller.get_export("memory") {
        Some(Extern::Memory(m)) => m,
        _ => panic!("worker module must export 'memory'"),
    };
    let data = mem.data_mut(&mut *caller);
    let start = ptr as usize;
    data[start..start + bytes.len()].copy_from_slice(bytes);
}

// Layout in the worker module's linear memory (matches worker_net.c).
const HOST_OFF: i32 = 0; // target host string
const REQ_OFF: i32 = 64; // request payload
const RECV_OFF: i32 = 128; // receive buffer

#[derive(Default)]
struct CompState {
    hits: Arc<Mutex<u64>>,
}

#[test]
fn worker_thread_does_real_tcp_roundtrip_concurrently() {
    let engine = engine();

    // ---- REAL localhost TCP echo server on a background thread. ------------
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let echo_addr = listener.local_addr().unwrap();
    eprintln!("echo server listening on {echo_addr}");
    let conns_served = Arc::new(AtomicU64::new(0));
    let served = conns_served.clone();
    let echo = std::thread::spawn(move || {
        // Serve exactly two connections (worker + main), then exit.
        for _ in 0..2 {
            let (mut sock, _peer) = match listener.accept() {
                Ok(v) => v,
                Err(_) => break,
            };
            served.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(move || {
                let mut buf = [0u8; 1024];
                loop {
                    match sock.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            if sock.write_all(&buf[..n]).is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            });
        }
    });

    let net = SharedNet::default();

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

    // Barrier so the worker's real TCP round-trip and the main thread's
    // component canonical-ABI loop + own TCP round-trip are provably in-flight
    // at the same time (not run sequentially).
    let barrier = Arc::new(Barrier::new(2));

    const WORKER_PAYLOAD: &[u8] = b"WORKER-NET-PAYLOAD-12345";
    const MAIN_PAYLOAD: &[u8] = b"MAIN-NET-PAYLOAD-67890";

    let weng = engine.clone();
    let wnet = net.clone();
    let wbarrier = barrier.clone();
    let whost = echo_addr.ip().to_string();
    let wport = echo_addr.port();

    // ---- WORKER thread: OWN Store + OWN plain Linker with PLAIN tcp_* imports.
    let worker = std::thread::spawn(move || -> (i32, Vec<u8>) {
        let mut wstore = Store::new(
            &weng,
            WorkerState {
                net: wnet,
                label: "worker".to_string(),
            },
        );

        // OWN plain core Linker -- plain host networking handlers wired here via
        // func_wrap, backed by the SHARED registry. NOT the canonical ABI.
        let mut wlinker: Linker<WorkerState> = Linker::new(&weng);
        wlinker
            .func_wrap(
                "monocp",
                "tcp_connect",
                |mut caller: Caller<'_, WorkerState>,
                 host_ptr: i32,
                 host_len: i32,
                 port: i32| {
                    let host_bytes = read_guest(&mut caller, host_ptr, host_len);
                    let host = String::from_utf8_lossy(&host_bytes).to_string();
                    let net = caller.data().net.clone();
                    let label = caller.data().label.clone();
                    net.connect(&label, &host, port as u16)
                },
            )
            .unwrap();
        wlinker
            .func_wrap(
                "monocp",
                "tcp_write",
                |mut caller: Caller<'_, WorkerState>, fd: i32, ptr: i32, len: i32| {
                    let bytes = read_guest(&mut caller, ptr, len);
                    let net = caller.data().net.clone();
                    net.write(fd, &bytes)
                },
            )
            .unwrap();
        wlinker
            .func_wrap(
                "monocp",
                "tcp_read",
                |mut caller: Caller<'_, WorkerState>, fd: i32, ptr: i32, len: i32| {
                    let net = caller.data().net.clone();
                    let mut buf = vec![0u8; len as usize];
                    let n = net.read(fd, &mut buf);
                    if n > 0 {
                        write_guest(&mut caller, ptr, &buf[..n as usize]);
                    }
                    n
                },
            )
            .unwrap();
        wlinker
            .func_wrap(
                "monocp",
                "tcp_close",
                |caller: Caller<'_, WorkerState>, fd: i32| {
                    caller.data().net.clone().close(fd)
                },
            )
            .unwrap();

        let wmod = Module::new(&weng, WORKER_WASM).unwrap();
        let winst = wlinker.instantiate(&mut wstore, &wmod).unwrap();

        // Seed the worker module's OWN memory: target host string + payload.
        let mem = match winst.get_export(&mut wstore, "memory") {
            Some(Extern::Memory(m)) => m,
            _ => panic!("worker module must export memory"),
        };
        {
            let data = mem.data_mut(&mut wstore);
            data[HOST_OFF as usize..HOST_OFF as usize + whost.len()]
                .copy_from_slice(whost.as_bytes());
            data[REQ_OFF as usize..REQ_OFF as usize + WORKER_PAYLOAD.len()]
                .copy_from_slice(WORKER_PAYLOAD);
        }

        let entry = winst
            .get_typed_func::<(i32, i32, i32), (i32,)>(&mut wstore, "worker_entry")
            .unwrap();

        // Release at the same instant as the main thread.
        wbarrier.wait();

        // REAL TCP round-trip happens inside this call: tcp_connect ->
        // tcp_write -> tcp_read(s) -> tcp_close, all real std::net I/O.
        let (rc,) = entry
            .call(&mut wstore, (whost.len() as i32, WORKER_PAYLOAD.len() as i32, wport as i32))
            .unwrap();

        // Read the echo bytes back out of the worker module's OWN memory.
        let echo = if rc > 0 {
            let data = mem.data(&wstore);
            data[RECV_OFF as usize..RECV_OFF as usize + rc as usize].to_vec()
        } else {
            Vec::new()
        };
        (rc, echo)
    });

    // ---- MAIN thread proceeds CONCURRENTLY: drive the component canonical ABI
    // AND do its own REAL tcp round-trip through the SAME shared registry.
    const ITERS: i32 = 4000;
    let emit = cinstance
        .get_typed_func::<(i32,), ()>(&mut cstore, "emit")
        .unwrap();

    // Release at the same instant the worker starts.
    barrier.wait();

    // (1) component canonical-ABI lifted loop.
    emit.call(&mut cstore, (ITERS,)).unwrap();
    emit.post_return(&mut cstore).unwrap();

    // (2) main thread's OWN real tcp round-trip via the SAME shared registry.
    let main_fd = net.connect("main", &echo_addr.ip().to_string(), echo_addr.port());
    assert!(main_fd >= 0, "main thread connect failed");
    let mw = net.write(main_fd, MAIN_PAYLOAD);
    assert_eq!(mw, MAIN_PAYLOAD.len() as i32, "main thread write");
    let mut main_echo = Vec::new();
    let mut buf = [0u8; 256];
    while main_echo.len() < MAIN_PAYLOAD.len() {
        let n = net.read(main_fd, &mut buf);
        assert!(n > 0, "main thread read returned {n}");
        main_echo.extend_from_slice(&buf[..n as usize]);
    }
    net.close(main_fd);

    // Join the worker; capture its return code + echo bytes.
    let (rc, worker_echo) = worker.join().unwrap();

    // ---- VERIFY (c): no trap, worker_entry returned a non-negative byte count.
    eprintln!("worker_entry returned rc={rc} (>=0 == all real tcp_* host calls succeeded)");
    assert!(
        rc >= 0,
        "worker_entry must return >=0; negative encodes which tcp step failed \
         (-1xxx connect, -2xxx write, -3xxx read)"
    );

    // ---- VERIFY (a): the worker's round-tripped echo bytes match EXACTLY.
    eprintln!(
        "worker round-trip: sent {:?}, received {:?}",
        String::from_utf8_lossy(WORKER_PAYLOAD),
        String::from_utf8_lossy(&worker_echo)
    );
    assert_eq!(
        worker_echo, WORKER_PAYLOAD,
        "worker's REAL TCP echo bytes (read back from its OWN linear memory) must match byte-exact"
    );

    // main thread's echo matches too.
    eprintln!(
        "main round-trip: sent {:?}, received {:?}",
        String::from_utf8_lossy(MAIN_PAYLOAD),
        String::from_utf8_lossy(&main_echo)
    );
    assert_eq!(main_echo, MAIN_PAYLOAD, "main thread's REAL TCP echo bytes must match");

    // ---- VERIFY (b): the shared registry coherently saw BOTH connections.
    {
        let reg = net.0.lock().unwrap();
        eprintln!(
            "shared net registry connect_log = {:?}, next_fd reached {}",
            reg.connect_log, reg.next_fd
        );
        let worker_conns = reg.connect_log.iter().filter(|(l, _)| l == "worker").count();
        let main_conns = reg.connect_log.iter().filter(|(l, _)| l == "main").count();
        assert_eq!(worker_conns, 1, "shared registry saw exactly one worker connection");
        assert_eq!(main_conns, 1, "shared registry saw exactly one main connection");
        assert_eq!(reg.next_fd, 2, "shared registry handed out 2 fds across both threads");
    }
    let served = conns_served.load(Ordering::SeqCst);
    eprintln!("echo server accepted {served} connections");
    assert_eq!(served, 2, "the REAL echo server accepted both threads' connections");

    // ---- VERIFY (d): the component's canonical-ABI state is intact & untouched.
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
        "DELIVERABLE: a worker OS thread completed a REAL TCP round-trip via PLAIN host \
         imports (monocp.tcp_connect/tcp_write/tcp_read/tcp_close) on its OWN Store + OWN \
         plain Linker, hitting a SHARED Mutex-protected host connection registry, \
         CONCURRENTLY with the main thread's preview2 component canonical ABI AND the \
         main thread's own real TCP round-trip through the same registry. Echo bytes \
         verified byte-exact off the wire."
    );

    echo.join().unwrap();
}
