/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! OFI manager actor — per-process libfabric resource management.
//!
//! Mirrors the structure of `IbvManagerActor` but uses libfabric APIs.
//! The EFA provider in libfabric transparently handles same-node transfers
//! via SHM, so no special loopback detection is needed.

use std::collections::HashMap;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::OncePortHandle;
use hyperactor::RefClient;
use hyperactor::reference;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use super::OfiBuffer;
use super::domain::OfiDomain;
use super::endpoint::OfiEndpoint;
use super::endpoint::OfiMr;
use super::primitives::OfiConfig;
use super::primitives::OfiEndpointInfo;
use super::primitives::ofi_supported;
use crate::RdmaOp;
use crate::RdmaOpType;
use crate::RdmaTransportLevel;
use crate::backend::RdmaBackend;
use crate::local_memory::RdmaLocalMemory;
use crate::rdma_manager_actor::GetOfiActorRefClient;
use crate::rdma_manager_actor::RdmaManagerActor;
use crate::rdma_manager_actor::RdmaManagerMessageClient;

// ---------------------------------------------------------------------------
// Remote messages (serializable, cross-actor)
// ---------------------------------------------------------------------------

/// Messages handled by [`OfiManagerActor`] from remote actors.
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub enum OfiManagerMessage {
    /// Register a buffer and return its remote-accessible description.
    RequestBuffer {
        remote_buf_id: usize,
        #[reply]
        reply: reference::OncePortRef<Option<OfiBuffer>>,
    },
    /// Release a buffer registration (fire-and-forget).
    ReleaseBuffer { remote_buf_id: usize },
    /// Get this endpoint's address for peer exchange.
    GetEndpointAddr {
        #[reply]
        reply: reference::OncePortRef<OfiEndpointInfo>,
    },
    /// (Messaging path) Post fi_recv into a registered buffer.
    ///
    /// The remote side posts a receive buffer so the caller can fi_send
    /// data into it. Used on providers without RMA (e.g., P4d / Nitro v3).
    PostRecv {
        buf_id: usize,
        size: usize,
        sender: reference::ActorRef<OfiManagerActor>,
        timeout_ms: u64,
        #[reply]
        reply: reference::OncePortRef<Result<(), String>>,
    },
    /// (Messaging path) fi_send data from a registered buffer to the caller.
    ///
    /// The remote side reads from its own buffer and sends it via fi_send
    /// to the caller. Used for ReadIntoLocal on non-RMA providers.
    SendFromBuffer {
        buf_id: usize,
        size: usize,
        dest: reference::ActorRef<OfiManagerActor>,
        timeout_ms: u64,
        #[reply]
        reply: reference::OncePortRef<Result<(), String>>,
    },
}
wirevalue::register_type!(OfiManagerMessage);

// ---------------------------------------------------------------------------
// Local messages (non-serializable, same-process only)
// ---------------------------------------------------------------------------

/// Local-only messages for operations that must run on the actor owning
/// the OFI endpoint (fi_write/fi_read/fi_send/CQ poll are not thread-safe).
#[derive(Handler, HandleClient, Debug)]
pub enum OfiManagerLocalMessage {
    /// Execute an RDMA operation (read or write) through the OFI endpoint.
    ExecuteOp {
        op_type: RdmaOpType,
        local_addr: usize,
        local_size: usize,
        remote_ofi_mgr: reference::ActorRef<OfiManagerActor>,
        remote_buffer: OfiBuffer,
        remote_buf_id: usize,
        timeout_ms: u64,
        #[reply]
        reply: OncePortHandle<Result<(), String>>,
    },
}

// ---------------------------------------------------------------------------
// Actor definition
// ---------------------------------------------------------------------------

/// Guard that stops the background CQ progress thread on drop.
/// Signals the cancel flag and joins the thread to ensure it's
/// fully stopped before the CQ/domain resources are freed.
struct CqProgressGuard {
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// When > 0, the background thread skips fi_cq_read so the main
    /// operation thread can read completions without them being stolen.
    active_ops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for CqProgressGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CqProgressGuard").finish()
    }
}

impl Drop for CqProgressGuard {
    fn drop(&mut self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Manages all libfabric resources for a single process.
#[derive(Debug)]
#[hyperactor::export(
    handlers = [OfiManagerMessage],
)]
pub struct OfiManagerActor {
    owner: std::sync::OnceLock<ActorHandle<RdmaManagerActor>>,

    /// Background CQ progress thread. Declared BEFORE domain/endpoint
    /// so it's dropped first (Rust drops fields in declaration order),
    /// ensuring the thread stops before the CQ is closed.
    _progress_guard: CqProgressGuard,

    domain: OfiDomain,
    endpoint: OfiEndpoint,

    /// Map from buffer_id to (MR handle, OfiBuffer description).
    buffer_registrations: HashMap<usize, (OfiMr, OfiBuffer)>,

    /// Next MR key to use for registration.
    next_mr_key: u64,

    /// Cached peer addresses: ActorId → fi_addr_t.
    peer_addrs: HashMap<reference::ActorId, libfabric_sys::fi_addr_t>,

    config: OfiConfig,
}

#[async_trait]
impl Actor for OfiManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        let rdma_handle = RdmaManagerActor::local_handle(this);
        self.owner
            .set(rdma_handle)
            .map_err(|_| anyhow::anyhow!("OfiManagerActor owner already set"))?;
        Ok(())
    }
}

/// RAII guard that pauses the background CQ progress thread while
/// the main thread is posting operations and polling completions.
struct ActiveOpsGuard(std::sync::Arc<std::sync::atomic::AtomicUsize>);
impl Drop for ActiveOpsGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl OfiManagerActor {
    /// Construct a new `OfiManagerActor`.
    pub fn new(config: Option<OfiConfig>) -> Result<Self> {
        if !ofi_supported() {
            anyhow::bail!("libfabric is not available on this system");
        }

        let config = config.unwrap_or_default();
        let domain = OfiDomain::new(&config)?;
        let endpoint = OfiEndpoint::new(&domain)?;

        // Spawn a background thread that continuously drives CQ progress.
        // This is needed for cross-node transfers: the remote peer's
        // libfabric layer sends control messages (handshake, read requests)
        // that must be processed by polling fi_cq_read. Without this thread,
        // cross-node operations time out because nobody processes incoming
        // messages on the server side.
        //
        // This is the same approach NCCL's aws-ofi-nccl plugin uses.
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let active_ops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cancel_clone = cancel.clone();
        let active_ops_clone = active_ops.clone();
        let cq_ptr = domain.cq as usize; // pass as usize to avoid Send issues with raw ptr
        let progress_thread = std::thread::Builder::new()
            .name("ofi-cq-progress".to_string())
            .spawn(move || {
                let cq = cq_ptr as *mut libfabric_sys::fid_cq;
                while !cancel_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    // Only poll CQ when no active operations are running.
                    // When an operation IS active, the main thread in execute_op_impl
                    // handles all CQ reads. If we also polled here, we'd steal the
                    // completion entry (CQ entries can only be read once) and the
                    // main thread would time out waiting for a completion that was
                    // already consumed.
                    if active_ops_clone.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                        let mut entry: libfabric_sys::fi_cq_data_entry =
                            unsafe { std::mem::zeroed() };
                        unsafe {
                            libfabric_sys::libfabric_sys_fi_cq_read(
                                cq,
                                &mut entry as *mut _ as *mut _,
                                1,
                            );
                        }
                    }
                    // Sleep briefly to avoid spinning at 100% CPU.
                    // 100μs gives ~10,000 polls/sec which is responsive
                    // enough for network handshakes while being gentle on CPU.
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            })
            .ok();

        tracing::info!("OFI backend initialized (provider: {:?})", config.provider);

        Ok(Self {
            owner: std::sync::OnceLock::new(),
            _progress_guard: CqProgressGuard {
                cancel,
                active_ops,
                thread: progress_thread,
            },
            domain,
            endpoint,
            buffer_registrations: HashMap::new(),
            next_mr_key: 1,
            peer_addrs: HashMap::new(),
            config,
        })
    }

    /// Get or insert a peer's fi_addr_t by exchanging endpoint addresses.
    ///
    /// When the remote peer is the same actor (same-process loopback),
    /// resolves the local address directly to avoid a deadlock — hyperactor
    /// processes messages one at a time per actor, so sending a message
    /// from an actor to itself during handler execution would block forever.
    async fn resolve_peer(
        &mut self,
        cx: &Context<'_, Self>,
        remote: &reference::ActorRef<OfiManagerActor>,
    ) -> Result<libfabric_sys::fi_addr_t> {
        let peer_id = remote.actor_id().clone();
        if let Some(&addr) = self.peer_addrs.get(&peer_id) {
            tracing::info!(%peer_id, fi_addr = addr, "peer address cache hit");
            return Ok(addr);
        }

        // Detect same-actor loopback: if the remote is us, use local
        // endpoint address directly instead of sending a message (which
        // would deadlock since we're already processing ExecuteOp).
        // Same pattern as ibverbs manager_actor.rs line 666.
        let self_ref: reference::ActorRef<OfiManagerActor> = cx.bind();
        let remote_info = if remote.actor_id() == self_ref.actor_id() {
            tracing::info!(%peer_id, "same-actor loopback, using local endpoint address");
            self.endpoint.get_name()?
        } else {
            tracing::info!(%peer_id, "peer address cache miss, exchanging addresses");
            remote.get_endpoint_addr(cx).await?
        };

        let fi_addr = self.endpoint.av_insert(&self.domain, &remote_info)?;
        self.peer_addrs.insert(peer_id, fi_addr);
        Ok(fi_addr)
    }

    /// Register a local memory region and return the MR + key.
    ///
    /// When the domain was negotiated with HMEM, all registrations must go
    /// through `fi_mr_regattr` — even for CPU memory (`FI_HMEM_SYSTEM`).
    /// GPU memory uses `FI_HMEM_CUDA` with the device ordinal.
    fn register_local_mr(&mut self, addr: usize, size: usize) -> Result<OfiMr> {
        let key = self.next_mr_key;
        self.next_mr_key += 1;

        if self.domain.hmem_supported {
            if crate::local_memory::is_device_ptr(addr) {
                let ordinal = Self::cuda_device_ordinal(addr)?;
                tracing::debug!(
                    addr,
                    size,
                    key,
                    ordinal,
                    "registering GPU MR via fi_mr_regattr (FI_HMEM_CUDA)",
                );
                self.endpoint.register_mr_hmem(
                    &self.domain,
                    addr,
                    size,
                    key,
                    libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA,
                    ordinal,
                )
            } else {
                tracing::debug!(
                    addr,
                    size,
                    key,
                    "registering CPU MR via fi_mr_regattr (FI_HMEM_SYSTEM)"
                );
                self.endpoint.register_mr_hmem(
                    &self.domain,
                    addr,
                    size,
                    key,
                    libfabric_sys::fi_hmem_iface_FI_HMEM_SYSTEM,
                    0,
                )
            }
        } else {
            tracing::debug!(addr, size, key, "registering CPU MR via fi_mr_reg");
            self.endpoint.register_mr(&self.domain, addr, size, key)
        }
    }

    /// Extract the CUDA device ordinal for a GPU pointer.
    fn cuda_device_ordinal(addr: usize) -> Result<i32> {
        let mut ordinal: i32 = -1;
        let err = unsafe {
            rdmaxcel_sys::rdmaxcel_cuPointerGetAttribute(
                &mut ordinal as *mut _ as *mut std::ffi::c_void,
                rdmaxcel_sys::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                addr as rdmaxcel_sys::CUdeviceptr,
            )
        };
        if err != rdmaxcel_sys::CUDA_SUCCESS || ordinal < 0 {
            anyhow::bail!(
                "failed to get CUDA device ordinal for addr {:#x}: err={}",
                addr,
                err,
            );
        }
        Ok(ordinal)
    }

    // -------------------------------------------------------------------
    // Shared helper: post operation + EAGAIN retry + CQ poll
    // -------------------------------------------------------------------

    /// Pause the CQ progress thread and return an RAII guard.
    fn pause_cq_progress(&self) -> ActiveOpsGuard {
        self._progress_guard
            .active_ops
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        ActiveOpsGuard(self._progress_guard.active_ops.clone())
    }

    /// Post an operation with EAGAIN retry, then poll CQ for one completion.
    ///
    /// This is the shared core for fi_write, fi_send, etc. The `post_fn`
    /// closure is called to post the operation. On EAGAIN, CQ progress is
    /// driven and tokio yields before retrying. After a successful post,
    /// the CQ is polled until one completion arrives or timeout.
    async fn post_and_poll_cq(
        &self,
        post_fn: impl Fn() -> Result<()>,
        timeout: Duration,
        label: &str,
    ) -> Result<()> {
        let op_start = std::time::Instant::now();
        let _ops_guard = self.pause_cq_progress();

        // Post with EAGAIN retry
        let mut eagain_retries: u64 = 0;
        let deadline = std::time::Instant::now() + timeout;
        loop {
            match post_fn() {
                Ok(()) => {
                    tracing::info!(
                        elapsed_us = op_start.elapsed().as_micros() as u64,
                        eagain_retries,
                        label,
                        "OFI op posted successfully",
                    );
                    break;
                }
                Err(e)
                    if format!("{}", e).contains("-11") && std::time::Instant::now() < deadline =>
                {
                    eagain_retries += 1;
                    {
                        let mut dummy: libfabric_sys::fi_cq_data_entry =
                            unsafe { std::mem::zeroed() };
                        unsafe {
                            libfabric_sys::libfabric_sys_fi_cq_read(
                                self.domain.cq,
                                &mut dummy as *mut _ as *mut _,
                                1,
                            );
                        }
                    }
                    tokio::task::yield_now().await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        // Poll CQ for one completion
        self.poll_cq_async(timeout).await?;

        tracing::info!(
            elapsed_us = op_start.elapsed().as_micros() as u64,
            label,
            "OFI op completed",
        );
        Ok(())
    }

    /// Poll CQ for one completion asynchronously, yielding to tokio between polls.
    ///
    /// Caller must have paused the CQ progress thread (ActiveOpsGuard).
    async fn poll_cq_async(&self, timeout: Duration) -> Result<()> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let poll_result = {
                let mut entry: libfabric_sys::fi_cq_data_entry = unsafe { std::mem::zeroed() };
                let ret = unsafe {
                    libfabric_sys::libfabric_sys_fi_cq_read(
                        self.domain.cq,
                        &mut entry as *mut _ as *mut _,
                        1,
                    )
                };

                if ret > 0 {
                    Ok(true)
                } else if ret == -(libfabric_sys::FI_EAGAIN as isize) {
                    Ok(false)
                } else {
                    let mut err_entry: libfabric_sys::fi_cq_err_entry =
                        unsafe { std::mem::zeroed() };
                    unsafe {
                        libfabric_sys::libfabric_sys_fi_cq_readerr(
                            self.domain.cq,
                            &mut err_entry,
                            0,
                        );
                    }
                    tracing::error!(
                        cq_ret = ret,
                        err = err_entry.err,
                        prov_errno = err_entry.prov_errno,
                        "CQ error",
                    );
                    Err(anyhow::anyhow!(
                        "fi_cq_read error: ret={}, err={}, prov_errno={}",
                        ret,
                        err_entry.err,
                        err_entry.prov_errno,
                    ))
                }
            };

            match poll_result {
                Ok(true) => return Ok(()),
                Ok(false) => {
                    if std::time::Instant::now() >= deadline {
                        tracing::error!(
                            timeout_ms = timeout.as_millis() as u64,
                            "CQ poll timed out — no completion received",
                        );
                        anyhow::bail!("CQ poll timed out");
                    }
                    tokio::task::yield_now().await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    // -------------------------------------------------------------------
    // Main operation dispatch
    // -------------------------------------------------------------------

    /// Execute an RDMA op: register local MR, resolve peer, transfer data, poll CQ.
    ///
    /// Dispatches to the RMA path (fi_write/fi_read) or messaging path
    /// (fi_send/fi_recv) based on provider capabilities.
    async fn execute_op_impl(
        &mut self,
        cx: &Context<'_, Self>,
        op_type: RdmaOpType,
        local_addr: usize,
        local_size: usize,
        remote_mgr: &reference::ActorRef<OfiManagerActor>,
        remote_buf: &OfiBuffer,
        remote_buf_id: usize,
        timeout: Duration,
    ) -> Result<()> {
        let op_start = std::time::Instant::now();
        tracing::info!(
            ?op_type,
            local_addr,
            local_size,
            rma_supported = self.domain.rma_supported,
            remote_buf_id,
            timeout_ms = timeout.as_millis() as u64,
            "OFI op starting",
        );

        if self.domain.rma_supported {
            self.execute_op_rma(cx, op_type, local_addr, local_size, remote_mgr, remote_buf, timeout)
                .await
        } else {
            self.execute_op_msg(cx, op_type, local_addr, local_size, remote_mgr, remote_buf_id, timeout)
                .await
        }?;

        tracing::info!(
            ?op_type,
            local_size,
            elapsed_us = op_start.elapsed().as_micros() as u64,
            "OFI op completed",
        );
        Ok(())
    }

    // -------------------------------------------------------------------
    // RMA path (fi_write / fi_read) — Nitro v4+
    // -------------------------------------------------------------------

    async fn execute_op_rma(
        &mut self,
        cx: &Context<'_, Self>,
        op_type: RdmaOpType,
        local_addr: usize,
        local_size: usize,
        remote_mgr: &reference::ActorRef<OfiManagerActor>,
        remote_buf: &OfiBuffer,
        timeout: Duration,
    ) -> Result<()> {
        let local_mr = self.register_local_mr(local_addr, local_size)?;
        let fi_addr = self.resolve_peer(cx, remote_mgr).await?;

        match op_type {
            RdmaOpType::WriteFromLocal => {
                self.post_and_poll_cq(
                    || {
                        self.endpoint.write(
                            local_addr as u64,
                            local_size,
                            local_mr.desc(),
                            remote_buf.addr,
                            remote_buf.rkey,
                            fi_addr,
                            ptr::null_mut(),
                        )
                    },
                    timeout,
                    "rma_write",
                )
                .await
            }
            RdmaOpType::ReadIntoLocal => {
                self.post_and_poll_cq(
                    || {
                        self.endpoint.read(
                            local_addr as u64,
                            local_size,
                            local_mr.desc(),
                            remote_buf.addr,
                            remote_buf.rkey,
                            fi_addr,
                            ptr::null_mut(),
                        )
                    },
                    timeout,
                    "rma_read",
                )
                .await
            }
        }
    }

    // -------------------------------------------------------------------
    // Messaging path (fi_send / fi_recv) — P4d / Nitro v3
    // -------------------------------------------------------------------

    async fn execute_op_msg(
        &mut self,
        cx: &Context<'_, Self>,
        op_type: RdmaOpType,
        local_addr: usize,
        local_size: usize,
        remote_mgr: &reference::ActorRef<OfiManagerActor>,
        remote_buf_id: usize,
        timeout: Duration,
    ) -> Result<()> {
        let local_mr = self.register_local_mr(local_addr, local_size)?;
        let fi_addr = self.resolve_peer(cx, remote_mgr).await?;
        let self_ref: reference::ActorRef<OfiManagerActor> = cx.bind();
        let is_loopback = remote_mgr.actor_id() == self_ref.actor_id();
        let timeout_ms = timeout.as_millis() as u64;

        match op_type {
            RdmaOpType::WriteFromLocal => {
                // Sender pushes data via fi_send; remote posts fi_recv first.
                if is_loopback {
                    // Loopback: post fi_recv on our own buffer, then fi_send.
                    let (dst_mr, dst_buf) = self
                        .buffer_registrations
                        .get(&remote_buf_id)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "loopback WriteFromLocal: buffer {} not registered",
                                remote_buf_id,
                            )
                        })?;
                    self.endpoint
                        .recv(dst_buf.addr, local_size, dst_mr.desc(), fi_addr, ptr::null_mut())?;
                } else {
                    // Cross-actor: tell remote to post fi_recv into its buffer.
                    remote_mgr
                        .post_recv(cx, remote_buf_id, local_size, self_ref.clone(), timeout_ms)
                        .await?
                        .map_err(|e| anyhow::anyhow!(e))?;
                }

                // fi_send from local buffer + poll CQ for send completion.
                self.post_and_poll_cq(
                    || {
                        self.endpoint
                            .send(local_addr as u64, local_size, local_mr.desc(), fi_addr, ptr::null_mut())
                    },
                    timeout,
                    "msg_send",
                )
                .await
            }
            RdmaOpType::ReadIntoLocal => {
                // Post fi_recv locally, then ask remote to fi_send.
                self.endpoint
                    .recv(local_addr as u64, local_size, local_mr.desc(), fi_addr, ptr::null_mut())?;

                if is_loopback {
                    // Loopback: fi_send from our own buffer.
                    let (src_mr, src_buf) = self
                        .buffer_registrations
                        .get(&remote_buf_id)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "loopback ReadIntoLocal: buffer {} not registered",
                                remote_buf_id,
                            )
                        })?;
                    // fi_send + poll CQ for send completion.
                    self.post_and_poll_cq(
                        || {
                            self.endpoint.send(
                                src_buf.addr,
                                local_size,
                                src_mr.desc(),
                                fi_addr,
                                ptr::null_mut(),
                            )
                        },
                        timeout,
                        "msg_send_loopback",
                    )
                    .await?;
                } else {
                    // Cross-actor: tell remote to fi_send from its buffer.
                    remote_mgr
                        .send_from_buffer(cx, remote_buf_id, local_size, self_ref.clone(), timeout_ms)
                        .await?
                        .map_err(|e| anyhow::anyhow!(e))?;
                }

                // Poll CQ for recv completion.
                let _ops_guard = self.pause_cq_progress();
                self.poll_cq_async(timeout).await
            }
        }
    }

    /// Construct an `ActorHandle` for the local `OfiManagerActor`.
    pub async fn local_handle(
        client: &(impl hyperactor::context::Actor + Send + Sync),
    ) -> Result<ActorHandle<Self>, anyhow::Error> {
        let rdma_handle = RdmaManagerActor::local_handle(client);
        let ofi_ref = rdma_handle
            .get_ofi_actor_ref(client)
            .await?
            .ok_or_else(|| anyhow::anyhow!("local RdmaManagerActor has no OFI backend"))?;
        ofi_ref
            .downcast_handle(client)
            .ok_or_else(|| anyhow::anyhow!("OfiManagerActor is not in the local process"))
    }
}

// ---------------------------------------------------------------------------
// Remote message handlers
// ---------------------------------------------------------------------------

#[async_trait]
#[hyperactor::handle(OfiManagerMessage)]
impl OfiManagerMessageHandler for OfiManagerActor {
    async fn request_buffer(
        &mut self,
        cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<Option<OfiBuffer>, anyhow::Error> {
        if let Some((_, buf)) = self.buffer_registrations.get(&remote_buf_id) {
            return Ok(Some(buf.clone()));
        }

        let owner = self.owner.get().unwrap();
        let mem = match owner.request_local_memory(cx, remote_buf_id).await? {
            Some(mem) => mem,
            None => return Ok(None),
        };

        let key = self.next_mr_key;
        self.next_mr_key += 1;
        let mr = self
            .endpoint
            .register_mr(&self.domain, mem.addr(), mem.size(), key)?;

        let buf = OfiBuffer {
            rkey: mr.rkey,
            addr: mem.addr() as u64,
            size: mem.size(),
        };

        self.buffer_registrations
            .insert(remote_buf_id, (mr, buf.clone()));
        Ok(Some(buf))
    }

    async fn release_buffer(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<(), anyhow::Error> {
        self.buffer_registrations.remove(&remote_buf_id);
        Ok(())
    }

    async fn get_endpoint_addr(
        &mut self,
        _cx: &Context<Self>,
    ) -> Result<OfiEndpointInfo, anyhow::Error> {
        self.endpoint.get_name()
    }

    async fn post_recv(
        &mut self,
        cx: &Context<Self>,
        buf_id: usize,
        size: usize,
        sender: reference::ActorRef<OfiManagerActor>,
        _timeout_ms: u64,
    ) -> Result<Result<(), String>, anyhow::Error> {
        // Resolve peer first (may mutate peer_addrs), then look up buffer.
        let fi_addr = self.resolve_peer(cx, &sender).await?;

        let (mr, buf) = self.buffer_registrations.get(&buf_id).ok_or_else(|| {
            anyhow::anyhow!("PostRecv: buffer {} not registered", buf_id)
        })?;

        let recv_size = size.min(buf.size);
        tracing::info!(buf_id, recv_size, fi_addr, "posting fi_recv for incoming data");

        Ok(self
            .endpoint
            .recv(buf.addr, recv_size, mr.desc(), fi_addr, ptr::null_mut())
            .map_err(|e| e.to_string()))
    }

    async fn send_from_buffer(
        &mut self,
        cx: &Context<Self>,
        buf_id: usize,
        size: usize,
        dest: reference::ActorRef<OfiManagerActor>,
        timeout_ms: u64,
    ) -> Result<Result<(), String>, anyhow::Error> {
        // Resolve peer first (may mutate peer_addrs), then look up buffer.
        let fi_addr = self.resolve_peer(cx, &dest).await?;

        let (mr, buf) = self.buffer_registrations.get(&buf_id).ok_or_else(|| {
            anyhow::anyhow!("SendFromBuffer: buffer {} not registered", buf_id)
        })?;

        let send_size = size.min(buf.size);
        let send_addr = buf.addr;
        // Store desc as usize to avoid holding a non-Send *mut c_void across await.
        let send_desc = mr.desc() as usize;
        tracing::info!(buf_id, send_size, fi_addr, "fi_send from buffer to remote");

        Ok(self
            .post_and_poll_cq(
                || {
                    self.endpoint.send(
                        send_addr,
                        send_size,
                        send_desc as *mut std::ffi::c_void,
                        fi_addr,
                        ptr::null_mut(),
                    )
                },
                Duration::from_millis(timeout_ms),
                "msg_send_from_buffer",
            )
            .await
            .map_err(|e| e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Local message handlers
// ---------------------------------------------------------------------------

#[async_trait]
#[hyperactor::handle(OfiManagerLocalMessage)]
impl OfiManagerLocalMessageHandler for OfiManagerActor {
    async fn execute_op(
        &mut self,
        cx: &Context<Self>,
        op_type: RdmaOpType,
        local_addr: usize,
        local_size: usize,
        remote_ofi_mgr: reference::ActorRef<OfiManagerActor>,
        remote_buffer: OfiBuffer,
        remote_buf_id: usize,
        timeout_ms: u64,
    ) -> Result<Result<(), String>, anyhow::Error> {
        Ok(self
            .execute_op_impl(
                cx,
                op_type,
                local_addr,
                local_size,
                &remote_ofi_mgr,
                &remote_buffer,
                remote_buf_id,
                Duration::from_millis(timeout_ms),
            )
            .await
            .map_err(|e| e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// OfiBackend — implements RdmaBackend trait
// ---------------------------------------------------------------------------

/// Wrapper around `ActorHandle<OfiManagerActor>` implementing `RdmaBackend`.
#[derive(Debug, Clone)]
pub struct OfiBackend(pub ActorHandle<OfiManagerActor>);

impl std::ops::Deref for OfiBackend {
    type Target = ActorHandle<OfiManagerActor>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl RdmaBackend for OfiBackend {
    type TransportInfo = ();

    async fn submit(
        &mut self,
        cx: &(impl hyperactor::context::Actor + Send + Sync),
        ops: Vec<RdmaOp>,
        timeout: Duration,
    ) -> Result<()> {
        let deadline = std::time::Instant::now() + timeout;
        for op in ops {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                anyhow::bail!("OFI submit timed out");
            }

            // Resolve remote OFI buffer (lazy)
            let (remote_ofi_mgr, remote_ofi_buf) = op
                .remote
                .resolve_ofi(cx)
                .await
                .ok_or_else(|| anyhow::anyhow!("OFI backend not found for remote buffer"))??;

            // Execute via local actor message (endpoint is not thread-safe)
            let remote_buf_id = op.remote.id;
            self.execute_op(
                cx,
                op.op_type,
                op.local.addr(),
                op.local.size(),
                remote_ofi_mgr,
                remote_ofi_buf,
                remote_buf_id,
                remaining.as_millis() as u64,
            )
            .await?
            .map_err(|e| anyhow::anyhow!(e))?;
        }
        Ok(())
    }

    fn transport_level(&self) -> RdmaTransportLevel {
        RdmaTransportLevel::Nic
    }

    fn transport_info(&self) -> Option<Self::TransportInfo> {
        None
    }
}
