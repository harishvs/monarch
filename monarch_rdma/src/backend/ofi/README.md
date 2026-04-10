# OFI (libfabric) RDMA Backend

## What this does

This backend lets Monarch actors transfer data using AWS EFA network cards through the libfabric library. The main reason it exists: the ibverbs backend can't transfer data between two actors on the **same machine** over EFA. Libfabric handles this automatically by routing same-node traffic through shared memory.

## How it works

```
Actor A (writer)                     Actor B (reader)
     |                                    |
     v                                    v
OfiManagerActor                    OfiManagerActor
     |                                    |
     v                                    v
  fi_write() -----> libfabric -----> fi_read()
                       |
              EFA provider decides:
              same node? → shared memory
              diff node? → EFA NIC (SRD)
```

Each process has one `OfiManagerActor` that owns a libfabric domain, endpoint, and completion queue. When an actor wants to read or write remote memory, the manager registers the memory, resolves the peer address, performs the operation, and waits for completion.

## Files

| File | What it does |
|------|-------------|
| `primitives.rs` | Config struct, cached availability probe (`ofi_supported()`) |
| `domain.rs` | Sets up the connection to the network card (fabric + domain + address vector + completion queue). Negotiates GPU memory support (HMEM) with fallback. |
| `endpoint.rs` | The data path — register memory, write, read, poll for completion. Handles both CPU (`fi_mr_reg`) and GPU (`fi_mr_regattr` with `FI_HMEM_CUDA`) memory. |
| `manager_actor.rs` | Hyperactor integration — message handlers for buffer registration, peer address exchange, and RDMA operations. Includes loopback detection for same-process peers. |
| `ofi_tests.rs` | 10 unit tests covering domain, endpoint, MR registration, address exchange, CPU and GPU data transfer roundtrips. |

## Key design decisions

**EAGAIN retry**: EFA's RDM layer returns EAGAIN on the first `fi_write`/`fi_read` to a new peer while an internal handshake completes. The code retries in a loop, driving CQ progress between attempts.

**Same-process loopback**: When source and destination actors share a process, they share the same `OfiManagerActor`. The `resolve_peer` method detects this by comparing actor IDs and uses the local endpoint address directly, avoiding a deadlock where the actor would send a message to itself.

**HMEM fallback**: The domain tries to negotiate `FI_HMEM` capability for GPU memory. If the provider doesn't support it, it falls back to CPU-only mode. When HMEM is active, all memory registrations (including CPU) must go through `fi_mr_regattr`.

**No `FI_LOCAL_MR`**: The deprecated `FI_LOCAL_MR` mode flag was forcing the `efa-direct` provider which requires manual receive buffer management. Removing it lets libfabric select the standard EFA provider that handles RMA internally.

## Dependencies

- **libfabric-sys** (`/libfabric-sys/`): FFI bindings. Includes `wrapper.c` with plain-C wrappers for 18 static-inline libfabric functions that bindgen can't capture.
- **libfabric** runtime: `/opt/amazon/efa/` on AWS EFA instances, or system libfabric (`libfabric-dev`).

## Limitations

- GPU data transfers require GPUDirect RDMA support (p5 instances with A100/H100). On g5/L4, GPU MR registration works but data doesn't actually transfer through the NIC.
- The `ofi` Cargo feature must be enabled at build time. `setup.py` auto-detects libfabric headers.
