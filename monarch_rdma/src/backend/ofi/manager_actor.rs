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
    ReleaseBuffer {
        remote_buf_id: usize,
    },
    /// Get this endpoint's address for peer exchange.
    GetEndpointAddr {
        #[reply]
        reply: reference::OncePortRef<OfiEndpointInfo>,
    },
}
wirevalue::register_type!(OfiManagerMessage);

// ---------------------------------------------------------------------------
// Local messages (non-serializable, same-process only)
// ---------------------------------------------------------------------------

/// Local-only messages for operations that must run on the actor owning
/// the OFI endpoint (fi_write/fi_read/CQ poll are not thread-safe).
#[derive(Handler, HandleClient, Debug)]
pub enum OfiManagerLocalMessage {
    /// Execute an RDMA operation (read or write) through the OFI endpoint.
    ExecuteOp {
        op_type: RdmaOpType,
        local_addr: usize,
        local_size: usize,
        remote_ofi_mgr: reference::ActorRef<OfiManagerActor>,
        remote_buffer: OfiBuffer,
        timeout_ms: u64,
        #[reply]
        reply: OncePortHandle<Result<(), String>>,
    },
}

// ---------------------------------------------------------------------------
// Actor definition
// ---------------------------------------------------------------------------

/// Manages all libfabric resources for a single process.
#[derive(Debug)]
#[hyperactor::export(
    handlers = [OfiManagerMessage],
)]
pub struct OfiManagerActor {
    owner: std::sync::OnceLock<ActorHandle<RdmaManagerActor>>,

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
        self.owner.set(rdma_handle).map_err(|_| {
            anyhow::anyhow!("OfiManagerActor owner already set")
        })?;
        Ok(())
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

        tracing::info!("OFI backend initialized (provider: {:?})", config.provider);

        Ok(Self {
            owner: std::sync::OnceLock::new(),
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
            tracing::debug!(%peer_id, fi_addr = addr, "peer address cache hit");
            return Ok(addr);
        }

        // Detect same-actor loopback: if the remote is us, use local
        // endpoint address directly instead of sending a message (which
        // would deadlock since we're already processing ExecuteOp).
        // Same pattern as ibverbs manager_actor.rs line 666.
        let self_ref: reference::ActorRef<OfiManagerActor> = cx.bind();
        let remote_info = if remote.actor_id() == self_ref.actor_id() {
            tracing::debug!(%peer_id, "same-actor loopback, using local endpoint address");
            self.endpoint.get_name()?
        } else {
            tracing::debug!(%peer_id, "peer address cache miss, exchanging addresses");
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
                tracing::debug!(addr, size, key, "registering CPU MR via fi_mr_regattr (FI_HMEM_SYSTEM)");
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

    /// Execute an RDMA op: register local MR, resolve peer, fi_write/fi_read, poll CQ.
    async fn execute_op_impl(
        &mut self,
        cx: &Context<'_, Self>,
        op_type: RdmaOpType,
        local_addr: usize,
        local_size: usize,
        remote_mgr: &reference::ActorRef<OfiManagerActor>,
        remote_buf: &OfiBuffer,
        timeout: Duration,
    ) -> Result<()> {
        let op_start = std::time::Instant::now();
        tracing::debug!(
            ?op_type,
            local_addr,
            local_size,
            remote_addr = remote_buf.addr,
            remote_rkey = remote_buf.rkey,
            "OFI op starting",
        );

        // 1. Register local memory
        let local_mr = self.register_local_mr(local_addr, local_size)?;

        // 2. Resolve peer address
        let fi_addr = self.resolve_peer(cx, remote_mgr).await?;

        // 3. Perform RDMA operation
        let result = match op_type {
            RdmaOpType::WriteFromLocal => self.endpoint.write(
                local_addr as u64,
                local_size,
                local_mr.desc(),
                remote_buf.addr,
                remote_buf.rkey,
                fi_addr,
                ptr::null_mut(),
            ),
            RdmaOpType::ReadIntoLocal => self.endpoint.read(
                local_addr as u64,
                local_size,
                local_mr.desc(),
                remote_buf.addr,
                remote_buf.rkey,
                fi_addr,
                ptr::null_mut(),
            ),
        };

        if let Err(e) = result {
            // local_mr dropped here, deregisters MR
            return Err(e);
        }

        // 4. Poll CQ for completion
        self.endpoint.poll_cq(&self.domain, timeout)?;

        let elapsed = op_start.elapsed();
        tracing::debug!(
            ?op_type,
            local_size,
            elapsed_us = elapsed.as_micros() as u64,
            "OFI op completed",
        );

        // local_mr dropped here, deregisters MR
        Ok(())
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
        let mr = self.endpoint.register_mr(&self.domain, mem.addr(), mem.size(), key)?;

        let buf = OfiBuffer {
            rkey: mr.rkey,
            addr: mem.addr() as u64,
            size: mem.size(),
        };

        self.buffer_registrations.insert(remote_buf_id, (mr, buf.clone()));
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
            self.execute_op(
                cx,
                op.op_type,
                op.local.addr(),
                op.local.size(),
                remote_ofi_mgr,
                remote_ofi_buf,
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
