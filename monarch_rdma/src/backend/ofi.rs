/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! OFI (libfabric) backend for RDMA operations.
//!
//! Uses the OpenFabrics Interfaces library to perform RDMA transfers.
//! On AWS EFA, libfabric's EFA provider transparently routes same-node
//! traffic through shared memory (SHM), avoiding the `ibv_create_ah`
//! loopback limitation that causes issue #3376.

pub mod domain;
pub mod endpoint;
pub mod manager_actor;
#[cfg(test)]
mod ofi_tests;
pub mod primitives;

use std::sync::Arc;

use hyperactor::reference;
use manager_actor::OfiManagerActor;

use crate::RdmaOpType;
use crate::local_memory::RdmaLocalMemory;

/// Serializable description of a remote buffer registered via OFI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, typeuri::Named)]
pub struct OfiBuffer {
    /// Remote memory region key (from `fi_mr_key`).
    pub rkey: u64,
    /// Remote virtual address of the registered buffer.
    pub addr: u64,
    /// Size of the buffer in bytes.
    pub size: usize,
}

/// A single RDMA op for the [`OfiBackend`](manager_actor::OfiBackend).
#[derive(Debug)]
pub struct OfiOp {
    pub op_type: RdmaOpType,
    pub local_memory: Arc<dyn RdmaLocalMemory>,
    pub remote_buffer: OfiBuffer,
    pub remote_manager: reference::ActorRef<OfiManagerActor>,
}
