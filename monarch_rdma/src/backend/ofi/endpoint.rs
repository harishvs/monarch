/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! OFI endpoint — create, bind, enable, and perform RDMA and messaging ops.

use std::ptr;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;

use super::domain::OfiDomain;
use super::primitives::OfiEndpointInfo;

/// An active OFI endpoint bound to a domain's AV and CQ.
pub struct OfiEndpoint {
    pub ep: *mut libfabric_sys::fid_ep,
}

// Safety: same as OfiDomain — single-threaded actor access.
unsafe impl Send for OfiEndpoint {}
unsafe impl Sync for OfiEndpoint {}

impl std::fmt::Debug for OfiEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfiEndpoint").field("ep", &self.ep).finish()
    }
}

impl OfiEndpoint {
    /// Create and enable an endpoint on the given domain.
    ///
    /// Binds the endpoint to the domain's AV (for peer addressing) and
    /// CQ (for completion notifications), then enables it.
    pub fn new(domain: &OfiDomain) -> Result<Self> {
        unsafe {
            let mut ep: *mut libfabric_sys::fid_ep = ptr::null_mut();
            let ret = libfabric_sys::libfabric_sys_fi_endpoint(
                domain.domain,
                domain.info,
                &mut ep,
                ptr::null_mut(),
            );
            if ret != 0 {
                anyhow::bail!("fi_endpoint failed: {}", ret);
            }

            // Bind AV
            let ret = libfabric_sys::libfabric_sys_fi_ep_bind(ep, &mut (*domain.av).fid, 0);
            if ret != 0 {
                libfabric_sys::libfabric_sys_fi_close(&mut (*ep).fid);
                anyhow::bail!("fi_ep_bind(av) failed: {}", ret);
            }

            // Bind CQ for both send and recv completions
            let cq_flags = libfabric_sys::FI_TRANSMIT as u64 | libfabric_sys::FI_RECV as u64;
            let ret = libfabric_sys::libfabric_sys_fi_ep_bind(ep, &mut (*domain.cq).fid, cq_flags);
            if ret != 0 {
                libfabric_sys::libfabric_sys_fi_close(&mut (*ep).fid);
                anyhow::bail!("fi_ep_bind(cq) failed: {}", ret);
            }

            // Enable endpoint
            let ret = libfabric_sys::libfabric_sys_fi_enable(ep);
            if ret != 0 {
                libfabric_sys::libfabric_sys_fi_close(&mut (*ep).fid);
                anyhow::bail!("fi_enable failed: {}", ret);
            }

            Ok(OfiEndpoint { ep })
        }
    }

    /// Get the local endpoint address for peer exchange.
    pub fn get_name(&self) -> Result<OfiEndpointInfo> {
        unsafe {
            let mut addr_len: usize = 0;
            // First call to get required size
            libfabric_sys::libfabric_sys_fi_getname(
                &mut (*self.ep).fid,
                ptr::null_mut(),
                &mut addr_len,
            );

            let mut addr = vec![0u8; addr_len];
            let ret = libfabric_sys::libfabric_sys_fi_getname(
                &mut (*self.ep).fid,
                addr.as_mut_ptr() as *mut _,
                &mut addr_len,
            );
            if ret != 0 {
                anyhow::bail!("fi_getname failed: {}", ret);
            }

            Ok(OfiEndpointInfo { addr, addr_len })
        }
    }

    /// Insert a peer address into the domain's address vector.
    /// Returns the `fi_addr_t` handle used for subsequent operations.
    pub fn av_insert(
        &self,
        domain: &OfiDomain,
        peer: &OfiEndpointInfo,
    ) -> Result<libfabric_sys::fi_addr_t> {
        unsafe {
            let mut fi_addr: libfabric_sys::fi_addr_t = 0;
            let ret = libfabric_sys::libfabric_sys_fi_av_insert(
                domain.av,
                peer.addr.as_ptr() as *const _,
                1,
                &mut fi_addr,
                0,
                ptr::null_mut(),
            );
            if ret != 1 {
                anyhow::bail!("fi_av_insert failed: expected 1, got {}", ret);
            }
            Ok(fi_addr)
        }
    }

    /// Register a memory region for RDMA access.
    /// Returns the MR handle and the remote key.
    pub fn register_mr(
        &self,
        domain: &OfiDomain,
        addr: usize,
        size: usize,
        key: u64,
    ) -> Result<OfiMr> {
        unsafe {
            let mut mr: *mut libfabric_sys::fid_mr = ptr::null_mut();
            let ret = libfabric_sys::libfabric_sys_fi_mr_reg(
                domain.domain,
                addr as *const _,
                size,
                (libfabric_sys::FI_REMOTE_READ
                    | libfabric_sys::FI_REMOTE_WRITE
                    | libfabric_sys::FI_READ
                    | libfabric_sys::FI_WRITE) as u64,
                0,   // offset
                key, // requested_key
                0,   // flags
                &mut mr,
                ptr::null_mut(),
            );
            if ret != 0 {
                anyhow::bail!("fi_mr_reg failed: {}", ret);
            }

            let rkey = libfabric_sys::libfabric_sys_fi_mr_key(mr);
            Ok(OfiMr { mr, rkey })
        }
    }

    /// Register a GPU memory region for RDMA access using `fi_mr_regattr`.
    ///
    /// Uses the HMEM interface to register device memory (e.g., CUDA GPU
    /// buffers) with the appropriate device interface and ordinal. The domain
    /// must have been created with `FI_HMEM` capability.
    pub fn register_mr_hmem(
        &self,
        domain: &OfiDomain,
        addr: usize,
        size: usize,
        key: u64,
        iface: u32,
        device_ordinal: i32,
    ) -> Result<OfiMr> {
        unsafe {
            let iov = libfabric_sys::iovec {
                iov_base: addr as *mut _,
                iov_len: size,
            };

            let mut attr: libfabric_sys::fi_mr_attr = std::mem::zeroed();
            attr.__bindgen_anon_1.mr_iov = &iov;
            attr.iov_count = 1;
            attr.access = (libfabric_sys::FI_REMOTE_READ
                | libfabric_sys::FI_REMOTE_WRITE
                | libfabric_sys::FI_READ
                | libfabric_sys::FI_WRITE) as u64;
            attr.offset = 0;
            attr.requested_key = key;
            attr.iface = iface;
            attr.device.cuda = device_ordinal;

            let mut mr: *mut libfabric_sys::fid_mr = ptr::null_mut();
            let ret = libfabric_sys::libfabric_sys_fi_mr_regattr(domain.domain, &attr, 0, &mut mr);
            if ret != 0 {
                anyhow::bail!(
                    "fi_mr_regattr failed: {} (iface={}, device={})",
                    ret,
                    iface,
                    device_ordinal,
                );
            }

            let rkey = libfabric_sys::libfabric_sys_fi_mr_key(mr);
            Ok(OfiMr { mr, rkey })
        }
    }

    /// Perform an RDMA write (local → remote).
    pub fn write(
        &self,
        local_addr: u64,
        local_len: usize,
        local_mr_desc: *mut std::ffi::c_void,
        remote_addr: u64,
        remote_key: u64,
        peer: libfabric_sys::fi_addr_t,
        context: *mut std::ffi::c_void,
    ) -> Result<()> {
        unsafe {
            let ret = libfabric_sys::libfabric_sys_fi_write(
                self.ep,
                local_addr as *const _,
                local_len,
                local_mr_desc,
                peer,
                remote_addr,
                remote_key,
                context,
            );
            if ret != 0 {
                anyhow::bail!("fi_write failed: {}", ret);
            }
            Ok(())
        }
    }

    /// Send data to a peer via messaging (fi_send).
    ///
    /// Used on providers without RMA support (e.g., EFA on P4d / Nitro v3).
    /// The peer must have a matching fi_recv posted.
    pub fn send(
        &self,
        local_addr: u64,
        local_len: usize,
        local_mr_desc: *mut std::ffi::c_void,
        peer: libfabric_sys::fi_addr_t,
        context: *mut std::ffi::c_void,
    ) -> Result<()> {
        unsafe {
            let ret = libfabric_sys::libfabric_sys_fi_send(
                self.ep,
                local_addr as *const _,
                local_len,
                local_mr_desc,
                peer,
                context,
            );
            if ret != 0 {
                anyhow::bail!("fi_send failed: {}", ret);
            }
            Ok(())
        }
    }

    /// Post a receive buffer for incoming data from a peer (fi_recv).
    ///
    /// Used on providers without RMA support. Specify the expected
    /// `src_addr` (peer's fi_addr_t) to filter messages from that peer,
    /// or `FI_ADDR_UNSPEC` to accept from any.
    pub fn recv(
        &self,
        local_addr: u64,
        local_len: usize,
        local_mr_desc: *mut std::ffi::c_void,
        src_addr: libfabric_sys::fi_addr_t,
        context: *mut std::ffi::c_void,
    ) -> Result<()> {
        unsafe {
            let ret = libfabric_sys::libfabric_sys_fi_recv(
                self.ep,
                local_addr as *mut _,
                local_len,
                local_mr_desc,
                src_addr,
                context,
            );
            if ret != 0 {
                anyhow::bail!("fi_recv failed: {}", ret);
            }
            Ok(())
        }
    }

    /// Perform an RDMA read (remote → local).
    pub fn read(
        &self,
        local_addr: u64,
        local_len: usize,
        local_mr_desc: *mut std::ffi::c_void,
        remote_addr: u64,
        remote_key: u64,
        peer: libfabric_sys::fi_addr_t,
        context: *mut std::ffi::c_void,
    ) -> Result<()> {
        unsafe {
            let ret = libfabric_sys::libfabric_sys_fi_read(
                self.ep,
                local_addr as *mut _,
                local_len,
                local_mr_desc,
                peer,
                remote_addr,
                remote_key,
                context,
            );
            if ret != 0 {
                anyhow::bail!("fi_read failed: {}", ret);
            }
            Ok(())
        }
    }

    /// Poll the CQ for completions, blocking until at least one arrives
    /// or the timeout expires.
    pub fn poll_cq(&self, domain: &OfiDomain, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            let mut entry: libfabric_sys::fi_cq_data_entry = unsafe { std::mem::zeroed() };
            let ret = unsafe {
                libfabric_sys::libfabric_sys_fi_cq_read(
                    domain.cq,
                    &mut entry as *mut _ as *mut _,
                    1,
                )
            };

            if ret > 0 {
                return Ok(());
            } else if ret == -(libfabric_sys::FI_EAGAIN as isize) {
                // No completion yet — check timeout
                if Instant::now() >= deadline {
                    anyhow::bail!("CQ poll timed out");
                }
                std::thread::yield_now();
                continue;
            } else {
                // Error — read the error entry for details
                let mut err_entry: libfabric_sys::fi_cq_err_entry = unsafe { std::mem::zeroed() };
                let readerr_ret = unsafe {
                    libfabric_sys::libfabric_sys_fi_cq_readerr(domain.cq, &mut err_entry, 0)
                };

                let prov_str = if readerr_ret > 0 {
                    let buf = [0u8; 256];
                    unsafe {
                        let msg = libfabric_sys::libfabric_sys_fi_cq_strerror(
                            domain.cq,
                            err_entry.prov_errno,
                            err_entry.err_data,
                            buf.as_ptr() as *mut _,
                            buf.len(),
                        );
                        if !msg.is_null() {
                            std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned()
                        } else {
                            format!("prov_errno={}", err_entry.prov_errno)
                        }
                    }
                } else {
                    format!(
                        "prov_errno={} (fi_cq_readerr={})",
                        err_entry.prov_errno, readerr_ret
                    )
                };

                tracing::error!(
                    cq_ret = ret,
                    err = err_entry.err,
                    prov_errno = err_entry.prov_errno,
                    flags = err_entry.flags,
                    len = err_entry.len,
                    "CQ error: {}",
                    prov_str,
                );
                anyhow::bail!(
                    "fi_cq_read error: ret={}, err={}, {}, flags={:#x}, len={}",
                    ret,
                    err_entry.err,
                    prov_str,
                    err_entry.flags,
                    err_entry.len,
                );
            }
        }
    }
}

impl Drop for OfiEndpoint {
    fn drop(&mut self) {
        if !self.ep.is_null() {
            unsafe {
                libfabric_sys::libfabric_sys_fi_close(&mut (*self.ep).fid);
            }
        }
    }
}

/// Registered memory region handle.
pub struct OfiMr {
    pub mr: *mut libfabric_sys::fid_mr,
    pub rkey: u64,
}

impl std::fmt::Debug for OfiMr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfiMr")
            .field("mr", &self.mr)
            .field("rkey", &self.rkey)
            .finish()
    }
}

unsafe impl Send for OfiMr {}
unsafe impl Sync for OfiMr {}

impl OfiMr {
    /// Get the local descriptor for use in data transfer operations.
    pub fn desc(&self) -> *mut std::ffi::c_void {
        unsafe { libfabric_sys::libfabric_sys_fi_mr_desc(self.mr) }
    }
}

impl Drop for OfiMr {
    fn drop(&mut self) {
        if !self.mr.is_null() {
            unsafe {
                libfabric_sys::libfabric_sys_fi_close(&mut (*self.mr).fid);
            }
        }
    }
}
