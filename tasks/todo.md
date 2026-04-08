# libfabric/OFI RDMA Backend for AWS EFA

Design doc: `.claude/plans/delegated-inventing-hearth.md`

## Phase 1: FFI Foundation (`libfabric-sys` crate)
- [x] Create `libfabric-sys/Cargo.toml` with `links = "fabric"`, bindgen dep
- [x] Write `build.rs` (pkg-config probe, `/opt/amazon/efa/lib64` fallback, bindgen)
- [x] Generate bindings for `fabric.h`, `fi_domain.h`, `fi_endpoint.h`, `fi_rma.h`, `fi_cm.h`, `fi_eq.h`, `fi_errno.h`
- [x] Export key functions via bindgen allowlist: `fi_getinfo`, `fi_fabric`, `fi_domain`, `fi_endpoint`, `fi_cq_open`, `fi_av_open`, `fi_mr_reg`, `fi_mr_regattr`, `fi_write`, `fi_read`, `fi_cq_read`, `fi_av_insert`, `fi_getname`, `fi_enable`, `fi_close`, `fi_mr_key`
- [x] Safe probe helpers: `ofi_available()`, `efa_ofi_available()`, `version()`
- [x] Add `libfabric-sys` to workspace `Cargo.toml`
- [x] Verify compilation on EFA instance (`cargo build -p libfabric-sys`)

## Phase 2: Core Primitives (`monarch_rdma/src/backend/ofi/`)
- [x] `primitives.rs` -- `OfiConfig`, `ofi_supported()` (cached `fi_getinfo` probe), `OfiEndpointInfo`
- [x] `domain.rs` -- `OfiDomain` struct (fabric + domain + AV + CQ lifecycle, Drop impl)
- [x] `endpoint.rs` -- `OfiEndpoint` (create, bind, enable, `fi_write`/`fi_read` wrappers, CQ polling, `OfiMr` with Drop)
- [x] `ofi.rs` -- `OfiBuffer` (serializable), `OfiOp` structs
- [ ] Unit tests: domain creation, endpoint lifecycle, CPU MR registration

## Phase 3: Manager Actor + CPU Transfers
- [x] `manager_actor.rs` -- `OfiManagerActor` with hyperactor message handlers (`RequestBuffer`, `ReleaseBuffer`, `GetEndpointAddr`)
- [x] `OfiBackend` struct implementing `RdmaBackend` trait
- [x] Address exchange via actor messages (`fi_getname` → `GetEndpointAddr` → `fi_av_insert`)
- [x] MR registration/deregistration for CPU memory (via `register_local_mr`, MR dropped after op)
- [x] Buffer cache (`HashMap<usize, (OfiMr, OfiBuffer)>`) mirroring ibverbs pattern
- [x] Peer address cache (`HashMap<ActorId, fi_addr_t>`)
- [x] `ExecuteOp` local message — full data path: register MR → resolve peer → fi_write/fi_read → poll CQ → deregister MR
- [ ] Rust unit tests: CPU read/write round-trip between two actors

## Phase 4: Integration into Existing System
- [x] `backend.rs` -- Add `#[cfg(feature = "ofi")] pub mod ofi`, `Ofi(...)` variant to `RdmaRemoteBackendContext`, update Serialize/Deserialize
- [x] `rdma_components.rs` -- Add `Ofi(OfiBackend)` to `RdmaLocalBackend`, update `choose_backend()` (ibverbs → OFI → TCP), add `has_ofi_backend()`, `resolve_ofi()`
- [x] `rdma_manager_actor.rs` -- Add `ofi` field, `GetOfiActorRef` message + handler, update `new()`, `init()`, `request_buffer()`, `release_buffer()`
- [x] `config.rs` -- Add `RDMA_DISABLE_OFI` config attr
- [x] `Cargo.toml` -- Add `ofi` feature + `libfabric-sys` optional dep
- [x] `efa.rs` -- Add `is_efa_ofi_available()` via `fi_getinfo(provider="efa")`
- [x] `lib.rs` -- Feature-gated `pub use backend::ofi`, `ofi_supported()` helper
- [ ] Integration test: end-to-end OFI transfer between two processes
- [ ] Verify compilation on EFA instance (`cargo build -p monarch_rdma --features ofi`)

## Phase 5: CUDA/GPU Memory Support
- [ ] `fi_mr_regattr()` with `FI_HMEM_CUDA` and device ordinal
- [ ] Reuse `device_selection` infrastructure for CUDA→EFA device mapping
- [ ] Build-time detection of `FI_HMEM` capability (libfabric >= 1.18)
- [ ] GPU tensor transfer tests (read/write round-trip)

## Phase 6: Python API + Test Parametrization
- [x] `monarch_rdma/extension/lib.rs` -- Add `is_ofi_available_py()` PyO3 binding
- [x] `python/monarch/_src/rdma/rdma.py` -- Add `is_ofi_available()`, update `get_rdma_backend()` to return `"ofi"`
- [x] `python/monarch/rdma/__init__.py` -- Export `is_ofi_available`
- [x] `python/tests/test_rdma.py` -- Extend `RDMA_BACKENDS` with `"ofi"`, update `@rdma_backends` decorator with `_backend_config()` helper
- [ ] `python/tests/test_rdma_unit.py` -- OFI backend isolation config
- [ ] Verify all existing parametrized tests pass with OFI backend (needs EFA)

## Phase 7: Hardening
- [ ] Graceful fallback chain: OFI failure → ibverbs → TCP
- [ ] `fi_cq_readerr()` error recovery and logging
- [ ] Performance benchmark: `examples/rdma_pingpong.py` OFI vs ibverbs on EFA
- [ ] Tracing/metrics integration (operation latency, fallback events)
- [ ] CI gating: `ofi` feature only on EFA runners, `pytest.mark.skipif` elsewhere

## Review
_(to be filled after implementation)_

---

# GitHub Issue #3376: RDMABuffer.read_into() delivery timeout on same node (EKS/EFA)

**Why this matters:** On EKS clusters with EFA hardware (e.g. p4d.24xlarge), `RDMABuffer.read_into()` between two actors on the **same Kubernetes node** hangs with a "delivery timeout." The `IbvManagerActor` never responds to the `RequestBuffer` message. This blocks the weight-sync path for GRPO training (FSDP learner → vLLM generator), forcing a 65-second `torch.save`/`torch.load` fallback for 3GB models.

## Root Cause (confirmed)
EFA hardware fundamentally does not support same-device loopback RDMA:
- `ibv_create_ah()` fails when destination GID == local device GID (confirmed: `ibv_ud_pingpong -d rdmap16s27 localhost` → "Failed to create AH")
- `ibv_rc_pingpong -d rdmap16s27` → "Couldn't create QP" (EFA uses SRD, not RC — expected)
- The `is_loopback` check in `manager_actor.rs:666` only matches same-actor, not same-node-different-actor
- Two different actors on the same node take the cross-node handshake path → `rdmaxcel_efa_connect()` → `ibv_create_ah()` → fails → delivery timeout

## Solution
Instead of patching ibverbs, we implement a **libfabric (OFI) backend** (Phases 1-4 above). Libfabric's EFA provider transparently detects same-node peers by comparing GIDs and routes traffic through shared memory (SHM), avoiding the `ibv_create_ah` limitation entirely. Cross-node traffic goes through EFA SRD as normal.

## Regression Tests
- [x] `test_same_node_read_into_between_separate_actors` — parametrized (ibverbs + TCP), two actors on same host
- [x] `test_same_node_read_into_ibverbs_only` — ibverbs forced, no TCP fallback, 15s timeout
- [x] `test_cross_node_read_into_between_hosts` — two virtual hosts via `ProcessJob({"hosts": 2})`
- [x] `test_cross_node_read_into_ibverbs_only` — ibverbs forced, cross-node
- [x] `test_same_actor_same_device_loopback_read_into` — loopback path (same actor)
- [x] `test_same_actor_same_device_loopback_write_from` — loopback path (same actor)
