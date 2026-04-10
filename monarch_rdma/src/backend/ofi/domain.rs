/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! OFI domain management — fabric, domain, AV, and CQ lifecycle.
//!
//! An `OfiDomain` owns the full libfabric resource chain and closes
//! everything in the correct order on `Drop`.

use std::ffi::CString;
use std::ptr;

use anyhow::Result;

use super::primitives::OfiConfig;

/// Owns a libfabric fabric + domain + address vector + completion queue.
///
/// All pointers are non-null after construction. The `Drop` impl closes
/// them in reverse order (CQ → AV → domain → fabric → info).
pub struct OfiDomain {
    pub info: *mut libfabric_sys::fi_info,
    pub fabric: *mut libfabric_sys::fid_fabric,
    pub domain: *mut libfabric_sys::fid_domain,
    pub av: *mut libfabric_sys::fid_av,
    pub cq: *mut libfabric_sys::fid_cq,
}

// Safety: OfiDomain is only accessed from the OfiManagerActor which is
// single-threaded (hyperactor serializes all message handlers).
unsafe impl Send for OfiDomain {}
unsafe impl Sync for OfiDomain {}

impl std::fmt::Debug for OfiDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfiDomain")
            .field("info", &self.info)
            .field("fabric", &self.fabric)
            .field("domain", &self.domain)
            .field("av", &self.av)
            .field("cq", &self.cq)
            .finish()
    }
}

impl OfiDomain {
    /// Create a new OFI domain with the given config.
    ///
    /// Performs the full libfabric init sequence:
    /// `fi_getinfo` → `fi_fabric` → `fi_domain` → `fi_av_open` → `fi_cq_open`
    pub fn new(config: &OfiConfig) -> Result<Self> {
        unsafe {
            // Build hints
            let hints = libfabric_sys::fi_allocinfo();
            if hints.is_null() {
                anyhow::bail!("fi_allocinfo failed");
            }

            // Request RDM endpoint with RMA + MSG capabilities
            (*(*hints).ep_attr).type_ = libfabric_sys::fi_ep_type_FI_EP_RDM;
            (*hints).caps =
                libfabric_sys::FI_RMA as u64 | libfabric_sys::FI_MSG as u64;
            (*hints).mode = libfabric_sys::LIBFABRIC_SYS_FI_LOCAL_MR;

            // Set provider hint if specified
            if !config.provider.is_empty() {
                let prov = CString::new(config.provider.as_str())
                    .map_err(|e| anyhow::anyhow!("invalid provider name: {}", e))?;
                (*(*hints).fabric_attr).prov_name = prov.into_raw();
            }

            // fi_getinfo
            let mut info: *mut libfabric_sys::fi_info = ptr::null_mut();
            let ret = libfabric_sys::fi_getinfo(
                libfabric_sys::fi_version(),
                ptr::null(),
                ptr::null(),
                0,
                hints,
                &mut info,
            );
            libfabric_sys::fi_freeinfo(hints);

            if ret != 0 || info.is_null() {
                anyhow::bail!(
                    "fi_getinfo failed: {} (provider hint: {:?})",
                    ofi_strerror(ret),
                    config.provider,
                );
            }

            // fi_fabric
            let mut fabric: *mut libfabric_sys::fid_fabric = ptr::null_mut();
            let ret = libfabric_sys::fi_fabric(
                (*info).fabric_attr,
                &mut fabric,
                ptr::null_mut(),
            );
            if ret != 0 {
                libfabric_sys::fi_freeinfo(info);
                anyhow::bail!("fi_fabric failed: {}", ofi_strerror(ret));
            }

            // fi_domain
            let mut domain: *mut libfabric_sys::fid_domain = ptr::null_mut();
            let ret = libfabric_sys::fi_domain(
                fabric,
                info,
                &mut domain,
                ptr::null_mut(),
            );
            if ret != 0 {
                libfabric_sys::fi_close(&mut (*fabric).fid);
                libfabric_sys::fi_freeinfo(info);
                anyhow::bail!("fi_domain failed: {}", ofi_strerror(ret));
            }

            // fi_av_open (address vector for peer routing)
            let mut av_attr: libfabric_sys::fi_av_attr = std::mem::zeroed();
            av_attr.type_ = libfabric_sys::fi_av_type_FI_AV_UNSPEC;
            let mut av: *mut libfabric_sys::fid_av = ptr::null_mut();
            let ret = libfabric_sys::fi_av_open(
                domain,
                &mut av_attr,
                &mut av,
                ptr::null_mut(),
            );
            if ret != 0 {
                libfabric_sys::fi_close(&mut (*domain).fid);
                libfabric_sys::fi_close(&mut (*fabric).fid);
                libfabric_sys::fi_freeinfo(info);
                anyhow::bail!("fi_av_open failed: {}", ofi_strerror(ret));
            }

            // fi_cq_open (completion queue)
            let mut cq_attr: libfabric_sys::fi_cq_attr = std::mem::zeroed();
            cq_attr.size = (config.max_send_wr + config.max_recv_wr) as u64;
            cq_attr.format = libfabric_sys::fi_cq_format_FI_CQ_FORMAT_DATA;
            let mut cq: *mut libfabric_sys::fid_cq = ptr::null_mut();
            let ret = libfabric_sys::fi_cq_open(
                domain,
                &mut cq_attr,
                &mut cq,
                ptr::null_mut(),
            );
            if ret != 0 {
                libfabric_sys::fi_close(&mut (*av).fid);
                libfabric_sys::fi_close(&mut (*domain).fid);
                libfabric_sys::fi_close(&mut (*fabric).fid);
                libfabric_sys::fi_freeinfo(info);
                anyhow::bail!("fi_cq_open failed: {}", ofi_strerror(ret));
            }

            Ok(OfiDomain {
                info,
                fabric,
                domain,
                av,
                cq,
            })
        }
    }
}

impl Drop for OfiDomain {
    fn drop(&mut self) {
        unsafe {
            if !self.cq.is_null() {
                libfabric_sys::fi_close(&mut (*self.cq).fid);
            }
            if !self.av.is_null() {
                libfabric_sys::fi_close(&mut (*self.av).fid);
            }
            if !self.domain.is_null() {
                libfabric_sys::fi_close(&mut (*self.domain).fid);
            }
            if !self.fabric.is_null() {
                libfabric_sys::fi_close(&mut (*self.fabric).fid);
            }
            if !self.info.is_null() {
                libfabric_sys::fi_freeinfo(self.info);
            }
        }
    }
}

/// Convert a libfabric error code to a human-readable string.
fn ofi_strerror(err: i32) -> String {
    if err == 0 {
        return "success".to_string();
    }
    unsafe {
        let ptr = libfabric_sys::fi_strerror((-err) as _);
        if ptr.is_null() {
            format!("unknown error ({})", err)
        } else {
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}
