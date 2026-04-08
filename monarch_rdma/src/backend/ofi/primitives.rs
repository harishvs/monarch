/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Safe wrappers around libfabric primitives.
//!
//! Provides `OfiConfig`, availability probes, and serializable endpoint
//! address types used for peer exchange via hyperactor messages.

use std::sync::OnceLock;

/// Cached result of OFI availability probe.
static OFI_SUPPORTED: OnceLock<bool> = OnceLock::new();

/// Configuration for the OFI backend.
#[derive(Debug, Clone)]
pub struct OfiConfig {
    /// Provider name hint (e.g., "efa", "verbs", "tcp", or empty for auto).
    pub provider: String,
    /// Maximum number of outstanding send work requests.
    pub max_send_wr: usize,
    /// Maximum number of outstanding recv work requests.
    pub max_recv_wr: usize,
}

impl Default for OfiConfig {
    fn default() -> Self {
        Self {
            provider: String::new(), // auto-detect
            max_send_wr: 128,
            max_recv_wr: 128,
        }
    }
}

/// Whether libfabric is available with at least one usable provider.
/// Result is cached after the first call.
pub fn ofi_supported() -> bool {
    *OFI_SUPPORTED.get_or_init(|| libfabric_sys::ofi_available())
}

/// Serializable endpoint address for peer exchange.
///
/// Contains the raw address bytes from `fi_getname()`, which encode
/// the provider-specific addressing info (GID+QPN for EFA, IP:port
/// for tcp, etc.). Peers use `fi_av_insert()` with these bytes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, typeuri::Named)]
pub struct OfiEndpointInfo {
    /// Raw endpoint address from `fi_getname`.
    pub addr: Vec<u8>,
    /// Size of the address in bytes.
    pub addr_len: usize,
}
