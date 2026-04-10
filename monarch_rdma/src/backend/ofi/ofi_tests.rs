/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for the OFI (libfabric) backend.
//!
//! All tests self-gate on `ofi_supported()` so they skip gracefully
//! when libfabric is not installed (e.g., non-EFA CI runners).

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::domain::OfiDomain;
    use super::super::endpoint::OfiEndpoint;
    use super::super::primitives::OfiConfig;
    use super::super::primitives::ofi_supported;

    /// Skip helper — returns true if OFI is not available.
    fn skip_if_no_ofi() -> bool {
        if !ofi_supported() {
            eprintln!("Skipping test: OFI (libfabric) not available");
            true
        } else {
            false
        }
    }

    /// Register a CPU MR, using the HMEM path when the domain requires it.
    fn register_cpu_mr(
        ep: &OfiEndpoint,
        domain: &OfiDomain,
        addr: usize,
        size: usize,
        key: u64,
    ) -> super::super::endpoint::OfiMr {
        if domain.hmem_supported {
            ep.register_mr_hmem(domain, addr, size, key, libfabric_sys::fi_hmem_iface_FI_HMEM_SYSTEM, 0)
                .expect("register_mr_hmem (SYSTEM) failed")
        } else {
            ep.register_mr(domain, addr, size, key)
                .expect("register_mr failed")
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: Primitive unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ofi_domain_creation() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig::default();
        let domain = OfiDomain::new(&config).expect("OfiDomain::new should succeed");

        assert!(!domain.info.is_null(), "info should be non-null");
        assert!(!domain.fabric.is_null(), "fabric should be non-null");
        assert!(!domain.domain.is_null(), "domain should be non-null");
        assert!(!domain.av.is_null(), "av should be non-null");
        assert!(!domain.cq.is_null(), "cq should be non-null");

        // Drop cleans up — no crash
    }

    #[test]
    fn test_ofi_endpoint_lifecycle() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig::default();
        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let endpoint = OfiEndpoint::new(&domain).expect("endpoint creation failed");

        assert!(!endpoint.ep.is_null(), "endpoint should be non-null");

        let info = endpoint.get_name().expect("get_name should succeed");
        assert!(info.addr_len > 0, "endpoint address should have non-zero length");
        assert_eq!(
            info.addr.len(),
            info.addr_len,
            "addr vec length should match addr_len"
        );

        // Drop cleans up endpoint then domain — no crash
    }

    #[test]
    fn test_ofi_cpu_mr_registration() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig::default();
        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let endpoint = OfiEndpoint::new(&domain).expect("endpoint creation failed");

        // Allocate a CPU buffer and register it
        let mut buf = vec![0u8; 4096];
        let addr = buf.as_mut_ptr() as usize;
        let size = buf.len();

        let mr = endpoint
            .register_mr(&domain, addr, size, 1)
            .expect("register_mr should succeed");

        // rkey should be valid (non-zero for most providers, but 0 is technically
        // valid for some — just check we got an MR handle)
        assert!(!mr.mr.is_null(), "MR handle should be non-null");

        let desc = mr.desc();
        // desc may be null for some providers (e.g., tcp), so we only check it's callable
        let _ = desc;

        // Drop deregisters MR — no crash
    }

    #[test]
    fn test_ofi_av_insert_roundtrip() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig::default();

        // Create two independent domain+endpoint pairs
        let domain1 = OfiDomain::new(&config).expect("domain1 creation failed");
        let ep1 = OfiEndpoint::new(&domain1).expect("ep1 creation failed");

        let domain2 = OfiDomain::new(&config).expect("domain2 creation failed");
        let ep2 = OfiEndpoint::new(&domain2).expect("ep2 creation failed");

        // Exchange names
        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");

        // Insert peer addresses
        let fi_addr_2_on_1 = ep1
            .av_insert(&domain1, &info2)
            .expect("av_insert of ep2 into ep1 failed");
        let fi_addr_1_on_2 = ep2
            .av_insert(&domain2, &info1)
            .expect("av_insert of ep1 into ep2 failed");

        // fi_addr_t should not be FI_ADDR_UNSPEC (which is typically u64::MAX)
        assert_ne!(
            fi_addr_2_on_1,
            u64::MAX,
            "fi_addr should not be FI_ADDR_UNSPEC"
        );
        assert_ne!(
            fi_addr_1_on_2,
            u64::MAX,
            "fi_addr should not be FI_ADDR_UNSPEC"
        );
    }

    #[test]
    fn test_ofi_endpoint_drop_cleanup() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig::default();

        // Create and immediately drop multiple times — verify no resource leak or crash
        for _ in 0..5 {
            let domain = OfiDomain::new(&config).expect("domain creation failed");
            let ep = OfiEndpoint::new(&domain).expect("endpoint creation failed");

            // Register and immediately deregister an MR
            let mut buf = vec![0u8; 1024];
            let mr = ep
                .register_mr(&domain, buf.as_mut_ptr() as usize, buf.len(), 1)
                .expect("register_mr failed");

            drop(mr);
            drop(ep);
            drop(domain);
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3: CPU read/write round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ofi_cpu_write_roundtrip() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig { request_hmem: false, ..OfiConfig::default() };
        let buf_size = 4096;

        let mut src_buf = vec![42u8; buf_size];
        let mut dst_buf = vec![0u8; buf_size];

        // Shared domain with two endpoints — same-process peers share a
        // domain so the EFA RDM layer can route internally without a
        // cross-domain handshake.
        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let ep1 = OfiEndpoint::new(&domain).expect("ep1 failed");
        let ep2 = OfiEndpoint::new(&domain).expect("ep2 failed");

        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");

        let fi_addr2_on_ep1 = ep1.av_insert(&domain, &info2).expect("av_insert failed");
        let _fi_addr1_on_ep2 = ep2.av_insert(&domain, &info1).expect("av_insert failed");

        let src_mr = register_cpu_mr(&ep1, &domain, src_buf.as_mut_ptr() as usize, buf_size, 1);
        let dst_mr = register_cpu_mr(&ep2, &domain, dst_buf.as_mut_ptr() as usize, buf_size, 2);

        let mut ctx: libfabric_sys::fi_context2 = unsafe { std::mem::zeroed() };
        ep1.write(
            src_buf.as_ptr() as u64,
            buf_size,
            src_mr.desc(),
            dst_buf.as_ptr() as u64,
            dst_mr.rkey,
            fi_addr2_on_ep1,
            &mut ctx as *mut _ as *mut std::ffi::c_void,
        )
        .expect("fi_write failed");

        ep1.poll_cq(&domain, Duration::from_secs(10))
            .expect("CQ poll timed out or errored");

        assert_eq!(
            dst_buf, src_buf,
            "destination buffer should match source after RDMA write"
        );
    }

    #[test]
    fn test_ofi_cpu_read_roundtrip() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig { request_hmem: false, ..OfiConfig::default() };
        let buf_size = 4096;

        let mut remote_buf = vec![99u8; buf_size];
        let mut local_buf = vec![0u8; buf_size];

        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let ep1 = OfiEndpoint::new(&domain).expect("ep1 failed");
        let ep2 = OfiEndpoint::new(&domain).expect("ep2 failed");

        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");

        let fi_addr2_on_ep1 = ep1.av_insert(&domain, &info2).expect("av_insert failed");
        let _fi_addr1_on_ep2 = ep2.av_insert(&domain, &info1).expect("av_insert failed");

        let local_mr = register_cpu_mr(&ep1, &domain, local_buf.as_mut_ptr() as usize, buf_size, 1);
        let remote_mr = register_cpu_mr(&ep2, &domain, remote_buf.as_mut_ptr() as usize, buf_size, 2);

        let mut ctx: libfabric_sys::fi_context2 = unsafe { std::mem::zeroed() };
        ep1.read(
            local_buf.as_mut_ptr() as u64,
            buf_size,
            local_mr.desc(),
            remote_buf.as_ptr() as u64,
            remote_mr.rkey,
            fi_addr2_on_ep1,
            &mut ctx as *mut _ as *mut std::ffi::c_void,
        )
        .expect("fi_read failed");

        ep1.poll_cq(&domain, Duration::from_secs(10))
            .expect("CQ poll timed out or errored");

        assert_eq!(
            local_buf, remote_buf,
            "local buffer should match remote after RDMA read"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 5: GPU (CUDA) HMEM tests
    // -----------------------------------------------------------------------

    /// Returns true (skip) if OFI or CUDA is unavailable.
    fn skip_if_no_gpu_ofi() -> bool {
        if skip_if_no_ofi() {
            return true;
        }
        let cuda_ok = unsafe {
            rdmaxcel_sys::rdmaxcel_cuInit(0) == rdmaxcel_sys::CUDA_SUCCESS
        };
        if !cuda_ok {
            eprintln!("Skipping test: CUDA not available");
            return true;
        }
        false
    }

    #[test]
    fn test_ofi_gpu_mr_registration() {
        if skip_if_no_gpu_ofi() {
            return;
        }

        let config = OfiConfig::default();
        let domain = OfiDomain::new(&config).expect("domain creation failed");
        if !domain.hmem_supported {
            eprintln!("Skipping: provider does not support FI_HMEM");
            return;
        }
        let endpoint = OfiEndpoint::new(&domain).expect("endpoint creation failed");

        let allocator = crate::backend::cuda_test_utils::CudaAllocator::get();
        let gpu_buf = allocator.allocate(0, 4096);

        let mr = endpoint
            .register_mr_hmem(
                &domain,
                gpu_buf.ptr(),
                gpu_buf.size(),
                1,
                libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA,
                0,
            )
            .expect("register_mr_hmem should succeed");

        assert!(!mr.mr.is_null(), "MR handle should be non-null");
    }

    #[test]
    fn test_ofi_gpu_write_roundtrip() {
        if skip_if_no_gpu_ofi() {
            return;
        }

        let config = OfiConfig::default();
        let buf_size = 4096;

        let domain1 = OfiDomain::new(&config).expect("domain1 failed");
        if !domain1.hmem_supported {
            eprintln!("Skipping: provider does not support FI_HMEM");
            return;
        }
        let ep1 = OfiEndpoint::new(&domain1).expect("ep1 failed");
        let domain2 = OfiDomain::new(&config).expect("domain2 failed");
        let ep2 = OfiEndpoint::new(&domain2).expect("ep2 failed");

        let allocator = crate::backend::cuda_test_utils::CudaAllocator::get();
        let src_gpu = allocator.allocate(0, buf_size);
        let dst_gpu = allocator.allocate(0, buf_size);

        // Fill source with pattern
        let src_data = vec![42u8; buf_size];
        unsafe {
            rdmaxcel_sys::rdmaxcel_cuMemcpyHtoD_v2(
                src_gpu.ptr() as rdmaxcel_sys::CUdeviceptr,
                src_data.as_ptr() as *const _,
                src_data.len(),
            );
        }

        // Exchange addresses
        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");
        let fi_addr2 = ep1.av_insert(&domain1, &info2).expect("av_insert failed");
        let _fi_addr1 = ep2.av_insert(&domain2, &info1).expect("av_insert failed");

        let src_mr = ep1
            .register_mr_hmem(&domain1, src_gpu.ptr(), src_gpu.size(), 1, libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA, 0)
            .expect("src MR failed");
        let dst_mr = ep2
            .register_mr_hmem(&domain2, dst_gpu.ptr(), dst_gpu.size(), 2, libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA, 0)
            .expect("dst MR failed");

        let write_result = ep1.write(
            src_gpu.ptr() as u64, buf_size, src_mr.desc(),
            dst_gpu.ptr() as u64, dst_mr.rkey, fi_addr2,
            std::ptr::null_mut(),
        );
        if let Err(e) = &write_result {
            let msg = format!("{}", e);
            if msg.contains("-11") {
                eprintln!(
                    "Skipping: fi_write returned EAGAIN — CUDA P2P/dmabuf not supported \
                     by this EFA driver + CUDA combination"
                );
                return;
            }
        }
        write_result.expect("fi_write failed");

        ep1.poll_cq(&domain1, Duration::from_secs(10))
            .expect("CQ poll failed");

        let mut result = vec![0u8; buf_size];
        unsafe {
            rdmaxcel_sys::rdmaxcel_cuMemcpyDtoH_v2(
                result.as_mut_ptr() as *mut _,
                dst_gpu.ptr() as rdmaxcel_sys::CUdeviceptr,
                result.len(),
            );
        }
        assert_eq!(result, src_data, "GPU dst should match src after RDMA write");
    }

    #[test]
    fn test_ofi_gpu_read_roundtrip() {
        if skip_if_no_gpu_ofi() {
            return;
        }

        let config = OfiConfig::default();
        let buf_size = 4096;

        let domain1 = OfiDomain::new(&config).expect("domain1 failed");
        if !domain1.hmem_supported {
            eprintln!("Skipping: provider does not support FI_HMEM");
            return;
        }
        let ep1 = OfiEndpoint::new(&domain1).expect("ep1 failed");
        let domain2 = OfiDomain::new(&config).expect("domain2 failed");
        let ep2 = OfiEndpoint::new(&domain2).expect("ep2 failed");

        let allocator = crate::backend::cuda_test_utils::CudaAllocator::get();
        let remote_gpu = allocator.allocate(0, buf_size);
        let local_gpu = allocator.allocate(0, buf_size);

        // Fill remote with pattern
        let remote_data = vec![99u8; buf_size];
        unsafe {
            rdmaxcel_sys::rdmaxcel_cuMemcpyHtoD_v2(
                remote_gpu.ptr() as rdmaxcel_sys::CUdeviceptr,
                remote_data.as_ptr() as *const _,
                remote_data.len(),
            );
        }

        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");
        let fi_addr2 = ep1.av_insert(&domain1, &info2).expect("av_insert failed");
        let _fi_addr1 = ep2.av_insert(&domain2, &info1).expect("av_insert failed");

        let local_mr = ep1
            .register_mr_hmem(&domain1, local_gpu.ptr(), local_gpu.size(), 1, libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA, 0)
            .expect("local MR failed");
        let remote_mr = ep2
            .register_mr_hmem(&domain2, remote_gpu.ptr(), remote_gpu.size(), 2, libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA, 0)
            .expect("remote MR failed");

        let read_result = ep1.read(
            local_gpu.ptr() as u64, buf_size, local_mr.desc(),
            remote_gpu.ptr() as u64, remote_mr.rkey, fi_addr2,
            std::ptr::null_mut(),
        );
        if let Err(e) = &read_result {
            let msg = format!("{}", e);
            if msg.contains("-11") {
                eprintln!(
                    "Skipping: fi_read returned EAGAIN — CUDA P2P/dmabuf not supported \
                     by this EFA driver + CUDA combination"
                );
                return;
            }
        }
        read_result.expect("fi_read failed");

        ep1.poll_cq(&domain1, Duration::from_secs(10))
            .expect("CQ poll failed");

        let mut result = vec![0u8; buf_size];
        unsafe {
            rdmaxcel_sys::rdmaxcel_cuMemcpyDtoH_v2(
                result.as_mut_ptr() as *mut _,
                local_gpu.ptr() as rdmaxcel_sys::CUdeviceptr,
                result.len(),
            );
        }
        assert_eq!(result, remote_data, "GPU local should match remote after RDMA read");
    }
}
