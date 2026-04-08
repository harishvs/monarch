/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Raw FFI bindings to libfabric (OpenFabrics Interfaces).
//!
//! This crate provides auto-generated bindings to the libfabric C library,
//! which offers a portable, high-performance fabric communication API.
//! The EFA provider in libfabric transparently handles same-node transfers
//! via shared memory (SHM), solving the loopback limitation of raw ibverbs
//! on AWS EFA devices.
//!
//! # Safety
//!
//! All functions in the generated bindings are `unsafe`. Higher-level safe
//! wrappers should be built in `monarch_rdma::backend::ofi`.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// ---------------------------------------------------------------------------
// Safe helpers for probing libfabric availability
// ---------------------------------------------------------------------------

use std::ffi::CString;
use std::ptr;
use std::sync::OnceLock;

/// Equivalent to the `fi_allocinfo()` static inline helper in libfabric.
/// `fi_allocinfo()` is `static inline` so bindgen cannot generate it;
/// it simply calls `fi_dupinfo(NULL)`.
///
/// # Safety
/// Returns a heap-allocated `fi_info` that must be freed with `fi_freeinfo`.
pub unsafe fn fi_allocinfo() -> *mut fi_info {
    fi_dupinfo(ptr::null())
}

/// Cached result of libfabric availability probe.
static OFI_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check whether libfabric is available and has at least one usable provider.
///
/// The result is cached after the first call. Returns `true` if `fi_getinfo`
/// succeeds with default hints (any provider that supports RDM + RMA).
pub fn ofi_available() -> bool {
    *OFI_AVAILABLE.get_or_init(ofi_available_impl)
}

fn ofi_available_impl() -> bool {
    unsafe {
        let hints: *mut fi_info = fi_allocinfo();
        if hints.is_null() {
            return false;
        }

        // Request RDM endpoint with RMA capability (RDMA read/write)
        (*hints).ep_attr.as_mut().unwrap().type_ = fi_ep_type_FI_EP_RDM;
        (*hints).caps = FI_RMA as u64 | FI_MSG as u64;
        (*hints).mode = LIBFABRIC_SYS_FI_LOCAL_MR;

        let mut info: *mut fi_info = ptr::null_mut();
        let ret = fi_getinfo(
            fi_version(),
            ptr::null(),     // node
            ptr::null(),     // service
            0,               // flags
            hints,
            &mut info,
        );

        fi_freeinfo(hints);

        if ret == 0 && !info.is_null() {
            fi_freeinfo(info);
            true
        } else {
            false
        }
    }
}

/// Check whether the EFA provider specifically is available.
pub fn efa_ofi_available() -> bool {
    unsafe {
        let hints: *mut fi_info = fi_allocinfo();
        if hints.is_null() {
            return false;
        }

        (*hints).ep_attr.as_mut().unwrap().type_ = fi_ep_type_FI_EP_RDM;
        (*hints).caps = FI_RMA as u64 | FI_MSG as u64;

        let provider = CString::new("efa").unwrap();
        (*hints).fabric_attr.as_mut().unwrap().prov_name = provider.into_raw();

        let mut info: *mut fi_info = ptr::null_mut();
        let ret = fi_getinfo(
            fi_version(),
            ptr::null(),
            ptr::null(),
            0,
            hints,
            &mut info,
        );

        // fi_freeinfo will free prov_name since it was set via fi_allocinfo
        fi_freeinfo(hints);

        if ret == 0 && !info.is_null() {
            fi_freeinfo(info);
            true
        } else {
            false
        }
    }
}

/// Return the libfabric API version this crate was built against.
pub fn version() -> u32 {
    unsafe { fi_version() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fi_version() {
        let ver = version();
        let major = ver >> 16;
        let minor = ver & 0xffff;
        // libfabric 1.x or 2.x
        assert!(major >= 1, "unexpected libfabric version {}.{}", major, minor);
        println!("libfabric version: {}.{}", major, minor);
    }

    #[test]
    fn test_ofi_available() {
        let available = ofi_available();
        println!("OFI available: {}", available);
        // Don't assert — may not have providers on CI
    }

    #[test]
    fn test_efa_ofi_available() {
        let available = efa_ofi_available();
        println!("EFA OFI available: {}", available);
        // Don't assert — EFA only on p4d/p5 instances
    }
}
