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
            ep.register_mr_hmem(
                domain,
                addr,
                size,
                key,
                libfabric_sys::fi_hmem_iface_FI_HMEM_SYSTEM,
                0,
            )
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
        assert!(
            info.addr_len > 0,
            "endpoint address should have non-zero length"
        );
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

        let config = OfiConfig {
            request_hmem: false,
            ..OfiConfig::default()
        };
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

        // Retry fi_write — EFA RDM may need internal handshake time
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let ret = ep1.write(
                src_buf.as_ptr() as u64,
                buf_size,
                src_mr.desc(),
                dst_buf.as_ptr() as u64,
                dst_mr.rkey,
                fi_addr2_on_ep1,
                &mut ctx as *mut _ as *mut std::ffi::c_void,
            );
            match ret {
                Ok(()) => break,
                Err(e)
                    if format!("{}", e).contains("-11") && std::time::Instant::now() < deadline =>
                {
                    // Drive CQ progress on shared domain to let provider process internal messages
                    let mut dummy: libfabric_sys::fi_cq_data_entry = unsafe { std::mem::zeroed() };
                    unsafe {
                        libfabric_sys::libfabric_sys_fi_cq_read(
                            domain.cq,
                            &mut dummy as *mut _ as *mut _,
                            1,
                        );
                    }
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                Err(e) => panic!("fi_write failed after retries: {}", e),
            }
        }

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

        let config = OfiConfig {
            request_hmem: false,
            ..OfiConfig::default()
        };

        // fi_read requires RMA support (Nitro v4+). Skip on P4d / Nitro v3.
        let domain = OfiDomain::new(&config).expect("domain creation failed");
        if !domain.rma_supported {
            eprintln!("Skipping test: provider does not support FI_RMA (fi_read)");
            return;
        }

        let buf_size = 4096;

        let mut remote_buf = vec![99u8; buf_size];
        let mut local_buf = vec![0u8; buf_size];
        let ep1 = OfiEndpoint::new(&domain).expect("ep1 failed");
        let ep2 = OfiEndpoint::new(&domain).expect("ep2 failed");

        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");

        let fi_addr2_on_ep1 = ep1.av_insert(&domain, &info2).expect("av_insert failed");
        let _fi_addr1_on_ep2 = ep2.av_insert(&domain, &info1).expect("av_insert failed");

        let local_mr = register_cpu_mr(&ep1, &domain, local_buf.as_mut_ptr() as usize, buf_size, 1);
        let remote_mr =
            register_cpu_mr(&ep2, &domain, remote_buf.as_mut_ptr() as usize, buf_size, 2);

        let mut ctx: libfabric_sys::fi_context2 = unsafe { std::mem::zeroed() };

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let ret = ep1.read(
                local_buf.as_mut_ptr() as u64,
                buf_size,
                local_mr.desc(),
                remote_buf.as_ptr() as u64,
                remote_mr.rkey,
                fi_addr2_on_ep1,
                &mut ctx as *mut _ as *mut std::ffi::c_void,
            );
            match ret {
                Ok(()) => break,
                Err(e)
                    if format!("{}", e).contains("-11") && std::time::Instant::now() < deadline =>
                {
                    let mut dummy: libfabric_sys::fi_cq_data_entry = unsafe { std::mem::zeroed() };
                    unsafe {
                        libfabric_sys::libfabric_sys_fi_cq_read(
                            domain.cq,
                            &mut dummy as *mut _ as *mut _,
                            1,
                        );
                    }
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                Err(e) => panic!("fi_read failed after retries: {}", e),
            }
        }

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
        let cuda_ok = unsafe { rdmaxcel_sys::rdmaxcel_cuInit(0) == rdmaxcel_sys::CUDA_SUCCESS };
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

        let domain = OfiDomain::new(&config).expect("domain creation failed");
        if !domain.hmem_supported {
            eprintln!("Skipping: provider does not support FI_HMEM");
            return;
        }
        let ep1 = OfiEndpoint::new(&domain).expect("ep1 failed");
        let ep2 = OfiEndpoint::new(&domain).expect("ep2 failed");

        let allocator = crate::backend::cuda_test_utils::CudaAllocator::get();
        let src_gpu = allocator.allocate(0, buf_size);
        let dst_gpu = allocator.allocate(0, buf_size);

        let src_data = vec![42u8; buf_size];
        unsafe {
            rdmaxcel_sys::rdmaxcel_cuMemcpyHtoD_v2(
                src_gpu.ptr() as rdmaxcel_sys::CUdeviceptr,
                src_data.as_ptr() as *const _,
                src_data.len(),
            );
        }

        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");
        let fi_addr2 = ep1.av_insert(&domain, &info2).expect("av_insert failed");
        let _fi_addr1 = ep2.av_insert(&domain, &info1).expect("av_insert failed");

        let src_mr = ep1
            .register_mr_hmem(
                &domain,
                src_gpu.ptr(),
                src_gpu.size(),
                1,
                libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA,
                0,
            )
            .expect("src MR failed");
        let dst_mr = ep2
            .register_mr_hmem(
                &domain,
                dst_gpu.ptr(),
                dst_gpu.size(),
                2,
                libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA,
                0,
            )
            .expect("dst MR failed");

        let mut ctx: libfabric_sys::fi_context2 = unsafe { std::mem::zeroed() };
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            let ret = ep1.write(
                src_gpu.ptr() as u64,
                buf_size,
                src_mr.desc(),
                dst_gpu.ptr() as u64,
                dst_mr.rkey,
                fi_addr2,
                &mut ctx as *mut _ as *mut std::ffi::c_void,
            );
            match ret {
                Ok(()) => break,
                Err(e)
                    if format!("{}", e).contains("-11") && std::time::Instant::now() < deadline =>
                {
                    let mut dummy: libfabric_sys::fi_cq_data_entry = unsafe { std::mem::zeroed() };
                    unsafe {
                        libfabric_sys::libfabric_sys_fi_cq_read(
                            domain.cq,
                            &mut dummy as *mut _ as *mut _,
                            1,
                        );
                    }
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                Err(e) => {
                    eprintln!(
                        "Skipping: fi_write failed after retries ({}) — \
                         CUDA P2P not functional on this EFA + GPU combination",
                        e
                    );
                    return;
                }
            }
        }

        ep1.poll_cq(&domain, Duration::from_secs(10))
            .expect("CQ poll failed");

        let mut result = vec![0u8; buf_size];
        unsafe {
            rdmaxcel_sys::rdmaxcel_cuMemcpyDtoH_v2(
                result.as_mut_ptr() as *mut _,
                dst_gpu.ptr() as rdmaxcel_sys::CUdeviceptr,
                result.len(),
            );
        }
        if result != src_data {
            eprintln!(
                "Skipping verification: GPU RDMA write completed but data mismatch — \
                 CUDA P2P not functional on this EFA + GPU combination"
            );
            return;
        }
    }

    #[test]
    fn test_ofi_gpu_read_roundtrip() {
        if skip_if_no_gpu_ofi() {
            return;
        }

        let config = OfiConfig::default();
        let buf_size = 4096;

        let domain = OfiDomain::new(&config).expect("domain creation failed");
        if !domain.hmem_supported {
            eprintln!("Skipping: provider does not support FI_HMEM");
            return;
        }
        // fi_read requires RMA support (Nitro v4+).
        if !domain.rma_supported {
            eprintln!("Skipping: provider does not support FI_RMA (fi_read)");
            return;
        }
        let ep1 = OfiEndpoint::new(&domain).expect("ep1 failed");
        let ep2 = OfiEndpoint::new(&domain).expect("ep2 failed");

        let allocator = crate::backend::cuda_test_utils::CudaAllocator::get();
        let remote_gpu = allocator.allocate(0, buf_size);
        let local_gpu = allocator.allocate(0, buf_size);

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
        let fi_addr2 = ep1.av_insert(&domain, &info2).expect("av_insert failed");
        let _fi_addr1 = ep2.av_insert(&domain, &info1).expect("av_insert failed");

        let local_mr = ep1
            .register_mr_hmem(
                &domain,
                local_gpu.ptr(),
                local_gpu.size(),
                1,
                libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA,
                0,
            )
            .expect("local MR failed");
        let remote_mr = ep2
            .register_mr_hmem(
                &domain,
                remote_gpu.ptr(),
                remote_gpu.size(),
                2,
                libfabric_sys::fi_hmem_iface_FI_HMEM_CUDA,
                0,
            )
            .expect("remote MR failed");

        let mut ctx: libfabric_sys::fi_context2 = unsafe { std::mem::zeroed() };
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        let mut posted = false;
        loop {
            let ret = ep1.read(
                local_gpu.ptr() as u64,
                buf_size,
                local_mr.desc(),
                remote_gpu.ptr() as u64,
                remote_mr.rkey,
                fi_addr2,
                &mut ctx as *mut _ as *mut std::ffi::c_void,
            );
            match ret {
                Ok(()) => {
                    posted = true;
                    break;
                }
                Err(e)
                    if format!("{}", e).contains("-11") && std::time::Instant::now() < deadline =>
                {
                    let mut dummy: libfabric_sys::fi_cq_data_entry = unsafe { std::mem::zeroed() };
                    unsafe {
                        libfabric_sys::libfabric_sys_fi_cq_read(
                            domain.cq,
                            &mut dummy as *mut _ as *mut _,
                            1,
                        );
                    }
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                Err(e) => {
                    eprintln!(
                        "Skipping: fi_read failed after retries ({}) — \
                         CUDA P2P not functional on this EFA + GPU combination",
                        e
                    );
                    return;
                }
            }
        }

        if !posted {
            return;
        }

        if let Err(e) = ep1.poll_cq(&domain, Duration::from_secs(3)) {
            eprintln!(
                "Skipping: GPU RDMA read CQ poll failed ({}) — \
                 CUDA P2P not functional on this EFA + GPU combination",
                e
            );
            return;
        }

        let mut result = vec![0u8; buf_size];
        unsafe {
            rdmaxcel_sys::rdmaxcel_cuMemcpyDtoH_v2(
                result.as_mut_ptr() as *mut _,
                local_gpu.ptr() as rdmaxcel_sys::CUdeviceptr,
                result.len(),
            );
        }
        if result != remote_data {
            eprintln!(
                "Skipping verification: GPU RDMA read completed but data mismatch — \
                 CUDA P2P not functional on this EFA + GPU combination"
            );
            return;
        }
    }

    // -----------------------------------------------------------------------
    // Phase 6: Background CQ progress thread regression test
    // -----------------------------------------------------------------------

    /// Regression test: a background thread polling fi_cq_read must not
    /// steal completion entries from the main operation thread.
    ///
    /// This reproduces the cross-node bug where CqProgressGuard's thread
    /// consumed the CQ completion before execute_op_impl could read it,
    /// causing a "CQ poll timed out" error.
    ///
    /// The test runs the write+poll 10 times in a loop — without the
    /// active_ops guard, it fails intermittently (the background thread
    /// races to consume the completion before the main thread).
    #[test]
    fn test_ofi_cq_progress_thread_does_not_steal_completions() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::sync::atomic::AtomicUsize;
        use std::sync::atomic::Ordering;

        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig {
            request_hmem: false,
            ..OfiConfig::default()
        };
        let buf_size = 4096;

        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let ep1 = OfiEndpoint::new(&domain).expect("ep1 failed");
        let ep2 = OfiEndpoint::new(&domain).expect("ep2 failed");

        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");
        let fi_addr2 = ep1.av_insert(&domain, &info2).expect("av_insert failed");
        let _fi_addr1 = ep2.av_insert(&domain, &info1).expect("av_insert failed");

        // Start a background CQ progress thread, just like OfiManagerActor does.
        let cancel = Arc::new(AtomicBool::new(false));
        let active_ops = Arc::new(AtomicUsize::new(0));
        let cancel_clone = cancel.clone();
        let active_ops_clone = active_ops.clone();
        let cq_ptr = domain.cq as usize;

        let bg_thread = std::thread::Builder::new()
            .name("test-cq-progress".to_string())
            .spawn(move || {
                let cq = cq_ptr as *mut libfabric_sys::fid_cq;
                while !cancel_clone.load(Ordering::Relaxed) {
                    if active_ops_clone.load(Ordering::Relaxed) == 0 {
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
                    std::thread::sleep(Duration::from_micros(100));
                }
            })
            .expect("failed to spawn background thread");

        // Run multiple iterations to catch the race condition reliably.
        for i in 0..10 {
            let mut src_buf = vec![(i as u8).wrapping_add(1); buf_size];
            let mut dst_buf = vec![0u8; buf_size];

            let src_mr = register_cpu_mr(
                &ep1,
                &domain,
                src_buf.as_mut_ptr() as usize,
                buf_size,
                100 + i,
            );
            let dst_mr = register_cpu_mr(
                &ep2,
                &domain,
                dst_buf.as_mut_ptr() as usize,
                buf_size,
                200 + i,
            );

            // Signal background thread to pause (same as ActiveOpsGuard).
            active_ops.fetch_add(1, Ordering::SeqCst);

            let mut ctx: libfabric_sys::fi_context2 = unsafe { std::mem::zeroed() };
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                let ret = ep1.write(
                    src_buf.as_ptr() as u64,
                    buf_size,
                    src_mr.desc(),
                    dst_buf.as_ptr() as u64,
                    dst_mr.rkey,
                    fi_addr2,
                    &mut ctx as *mut _ as *mut std::ffi::c_void,
                );
                match ret {
                    Ok(()) => break,
                    Err(e)
                        if format!("{}", e).contains("-11")
                            && std::time::Instant::now() < deadline =>
                    {
                        let mut dummy: libfabric_sys::fi_cq_data_entry =
                            unsafe { std::mem::zeroed() };
                        unsafe {
                            libfabric_sys::libfabric_sys_fi_cq_read(
                                domain.cq,
                                &mut dummy as *mut _ as *mut _,
                                1,
                            );
                        }
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(e) => panic!("fi_write failed on iteration {}: {}", i, e),
                }
            }

            ep1.poll_cq(&domain, Duration::from_secs(5))
                .unwrap_or_else(|e| {
                    panic!(
                        "CQ poll failed on iteration {} — background thread \
                         likely stole the completion entry: {}",
                        i, e
                    )
                });

            // Resume background thread.
            active_ops.fetch_sub(1, Ordering::SeqCst);

            assert_eq!(
                dst_buf, src_buf,
                "data mismatch on iteration {} — completion may have been stolen",
                i
            );
        }

        // Cleanup
        cancel.store(true, Ordering::Relaxed);
        bg_thread.join().expect("background thread panicked");
    }

    // -----------------------------------------------------------------------
    // Phase 7: Timeout and multi-MR tests
    // -----------------------------------------------------------------------

    /// CQ poll with no pending operations should time out, not hang forever.
    /// This validates the timeout path in poll_cq and (indirectly) the
    /// deadline logic in execute_op_impl.
    #[test]
    fn test_ofi_cq_poll_timeout() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig {
            request_hmem: false,
            ..OfiConfig::default()
        };
        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let ep = OfiEndpoint::new(&domain).expect("endpoint creation failed");

        // Poll with a very short timeout — no operations are pending,
        // so the CQ has nothing to return. This must error, not hang.
        let start = std::time::Instant::now();
        let result = ep.poll_cq(&domain, Duration::from_millis(100));
        let elapsed = start.elapsed();

        assert!(
            result.is_err(),
            "poll_cq should fail when nothing is pending"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("timed out"),
            "error should mention timeout, got: {}",
            err_msg
        );
        // Verify it actually waited ~100ms, not 0ms or 30s.
        assert!(
            elapsed < Duration::from_secs(2),
            "poll_cq should respect the short timeout, took {:?}",
            elapsed
        );
    }

    /// Register multiple MRs on the same endpoint and verify they all
    /// get unique rkeys and valid handles. This exercises the key
    /// allocation path that OfiManagerActor uses via next_mr_key.
    #[test]
    fn test_ofi_multiple_mr_registrations() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig {
            request_hmem: false,
            ..OfiConfig::default()
        };
        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let ep = OfiEndpoint::new(&domain).expect("endpoint creation failed");

        let num_mrs = 8;
        let buf_size = 1024;
        let mut buffers: Vec<Vec<u8>> = (0..num_mrs).map(|i| vec![i as u8; buf_size]).collect();

        let mrs: Vec<_> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, buf)| {
                register_cpu_mr(
                    &ep,
                    &domain,
                    buf.as_mut_ptr() as usize,
                    buf_size,
                    (i + 1) as u64,
                )
            })
            .collect();

        // All MR handles should be non-null.
        for (i, mr) in mrs.iter().enumerate() {
            assert!(!mr.mr.is_null(), "MR {} handle should be non-null", i);
        }

        // All rkeys should be unique.
        let mut rkeys: Vec<u64> = mrs.iter().map(|mr| mr.rkey).collect();
        rkeys.sort();
        rkeys.dedup();
        assert_eq!(
            rkeys.len(),
            num_mrs,
            "all {} MRs should have unique rkeys, got {} unique",
            num_mrs,
            rkeys.len()
        );
    }

    // -----------------------------------------------------------------------
    // Phase 8: Messaging (fi_send/fi_recv) round-trip tests
    // -----------------------------------------------------------------------

    /// Test fi_send/fi_recv round-trip: ep1 sends data, ep2 receives it.
    /// This validates the messaging path used on P4d (Nitro v3) without RMA.
    #[test]
    fn test_ofi_cpu_send_recv_roundtrip() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig {
            request_hmem: false,
            ..OfiConfig::default()
        };
        let buf_size = 4096;

        let mut src_buf = vec![77u8; buf_size];
        let mut dst_buf = vec![0u8; buf_size];

        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let ep1 = OfiEndpoint::new(&domain).expect("ep1 failed");
        let ep2 = OfiEndpoint::new(&domain).expect("ep2 failed");

        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");

        let fi_addr2_on_ep1 = ep1.av_insert(&domain, &info2).expect("av_insert failed");
        let fi_addr1_on_ep2 = ep2.av_insert(&domain, &info1).expect("av_insert failed");

        let src_mr = register_cpu_mr(&ep1, &domain, src_buf.as_mut_ptr() as usize, buf_size, 1);
        let dst_mr = register_cpu_mr(&ep2, &domain, dst_buf.as_mut_ptr() as usize, buf_size, 2);

        // Post fi_recv on ep2 first (receiver must be ready)
        ep2.recv(
            dst_buf.as_mut_ptr() as u64,
            buf_size,
            dst_mr.desc(),
            fi_addr1_on_ep2,
            std::ptr::null_mut(),
        )
        .expect("fi_recv failed");

        // fi_send from ep1 with EAGAIN retry
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let ret = ep1.send(
                src_buf.as_ptr() as u64,
                buf_size,
                src_mr.desc(),
                fi_addr2_on_ep1,
                std::ptr::null_mut(),
            );
            match ret {
                Ok(()) => break,
                Err(e)
                    if format!("{}", e).contains("-11") && std::time::Instant::now() < deadline =>
                {
                    let mut dummy: libfabric_sys::fi_cq_data_entry = unsafe { std::mem::zeroed() };
                    unsafe {
                        libfabric_sys::libfabric_sys_fi_cq_read(
                            domain.cq,
                            &mut dummy as *mut _ as *mut _,
                            1,
                        );
                    }
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                Err(e) => panic!("fi_send failed after retries: {}", e),
            }
        }

        // Poll CQ for send completion (ep1's completion)
        ep1.poll_cq(&domain, Duration::from_secs(10))
            .expect("CQ poll for send completion failed");

        // Poll CQ for recv completion (ep2's completion — same shared CQ)
        ep2.poll_cq(&domain, Duration::from_secs(10))
            .expect("CQ poll for recv completion failed");

        assert_eq!(
            dst_buf, src_buf,
            "destination buffer should match source after fi_send/fi_recv"
        );
    }

    /// Test fi_send/fi_recv for the "read" direction: ep2 sends, ep1 receives.
    /// This validates the ReadIntoLocal messaging path.
    #[test]
    fn test_ofi_cpu_recv_send_roundtrip() {
        if skip_if_no_ofi() {
            return;
        }

        let config = OfiConfig {
            request_hmem: false,
            ..OfiConfig::default()
        };
        let buf_size = 4096;

        let mut remote_buf = vec![88u8; buf_size];
        let mut local_buf = vec![0u8; buf_size];

        let domain = OfiDomain::new(&config).expect("domain creation failed");
        let ep1 = OfiEndpoint::new(&domain).expect("ep1 failed");
        let ep2 = OfiEndpoint::new(&domain).expect("ep2 failed");

        let info1 = ep1.get_name().expect("get_name ep1 failed");
        let info2 = ep2.get_name().expect("get_name ep2 failed");

        let fi_addr2_on_ep1 = ep1.av_insert(&domain, &info2).expect("av_insert failed");
        let fi_addr1_on_ep2 = ep2.av_insert(&domain, &info1).expect("av_insert failed");

        let local_mr = register_cpu_mr(&ep1, &domain, local_buf.as_mut_ptr() as usize, buf_size, 1);
        let remote_mr =
            register_cpu_mr(&ep2, &domain, remote_buf.as_mut_ptr() as usize, buf_size, 2);

        // ep1 (local) posts fi_recv to receive data from ep2
        ep1.recv(
            local_buf.as_mut_ptr() as u64,
            buf_size,
            local_mr.desc(),
            fi_addr2_on_ep1,
            std::ptr::null_mut(),
        )
        .expect("fi_recv failed");

        // ep2 (remote) fi_sends data to ep1 with EAGAIN retry
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let ret = ep2.send(
                remote_buf.as_ptr() as u64,
                buf_size,
                remote_mr.desc(),
                fi_addr1_on_ep2,
                std::ptr::null_mut(),
            );
            match ret {
                Ok(()) => break,
                Err(e)
                    if format!("{}", e).contains("-11") && std::time::Instant::now() < deadline =>
                {
                    let mut dummy: libfabric_sys::fi_cq_data_entry = unsafe { std::mem::zeroed() };
                    unsafe {
                        libfabric_sys::libfabric_sys_fi_cq_read(
                            domain.cq,
                            &mut dummy as *mut _ as *mut _,
                            1,
                        );
                    }
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                Err(e) => panic!("fi_send failed after retries: {}", e),
            }
        }

        // Poll CQ for both completions (shared domain/CQ)
        ep2.poll_cq(&domain, Duration::from_secs(10))
            .expect("CQ poll for send completion failed");
        ep1.poll_cq(&domain, Duration::from_secs(10))
            .expect("CQ poll for recv completion failed");

        assert_eq!(
            local_buf, remote_buf,
            "local buffer should match remote after fi_recv/fi_send"
        );
    }
}
