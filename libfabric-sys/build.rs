/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Build script for libfabric-sys.
//!
//! Locates libfabric headers and library via:
//! 1. pkg-config (preferred)
//! 2. Common install paths (/opt/amazon/efa, /usr/local, /usr)
//!
//! Generates Rust FFI bindings using bindgen.

use std::env;
use std::path::PathBuf;

/// Well-known paths where libfabric may be installed.
const SEARCH_PATHS: &[&str] = &[
    "/opt/amazon/efa",  // AWS EFA installer default
    "/usr/local",       // Manual install default
    "/usr",             // System package default
];

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");

    let mut include_dir: Option<String> = None;
    let mut lib_dir: Option<String> = None;

    // Try pkg-config first
    if let Ok(lib) = pkg_config::Config::new()
        .atleast_version("1.10")
        .probe("libfabric")
    {
        // pkg-config found it — extract paths
        if let Some(dir) = lib.include_paths.first() {
            include_dir = Some(dir.to_string_lossy().into_owned());
        }
        if let Some(dir) = lib.link_paths.first() {
            lib_dir = Some(dir.to_string_lossy().into_owned());
        }
    } else {
        // Fallback: search well-known paths
        for base in SEARCH_PATHS {
            let inc = format!("{}/include", base);
            let header = format!("{}/rdma/fabric.h", inc);
            if std::path::Path::new(&header).exists() {
                include_dir = Some(inc);

                // Check lib64 first (EFA, RHEL), then lib
                let lib64 = format!("{}/lib64", base);
                let lib = format!("{}/lib", base);
                if std::path::Path::new(&format!("{}/libfabric.so", lib64)).exists() {
                    lib_dir = Some(lib64);
                } else if std::path::Path::new(&format!("{}/libfabric.so", lib)).exists() {
                    lib_dir = Some(lib);
                }
                break;
            }
        }
    }

    let include_dir = include_dir.unwrap_or_else(|| {
        panic!(
            "Could not find libfabric headers. Install libfabric-dev or set \
             LIBFABRIC_INCLUDE_DIR. Searched pkg-config and {:?}",
            SEARCH_PATHS
        )
    });

    if let Some(ref dir) = lib_dir {
        println!("cargo:rustc-link-search=native={}", dir);
    }
    println!("cargo:rustc-link-lib=dylib=fabric");

    // Export include dir for downstream crates
    println!("cargo:include={}", include_dir);

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir))
        // Only generate bindings for fi_* symbols and FI_* constants
        .allowlist_function("fi_.*")
        .allowlist_type("fi_.*")
        .allowlist_type("fid_.*")
        .allowlist_var("FI_.*")
        .allowlist_var("fi_.*")
        .allowlist_var("LIBFABRIC_SYS_.*")
        // Also allow the struct tags used by libfabric
        .allowlist_type("fi_info")
        .allowlist_type("fi_fabric_attr")
        .allowlist_type("fi_domain_attr")
        .allowlist_type("fi_ep_attr")
        .allowlist_type("fi_tx_attr")
        .allowlist_type("fi_rx_attr")
        .allowlist_type("fi_cq_attr")
        .allowlist_type("fi_av_attr")
        .allowlist_type("fi_mr_attr")
        .allowlist_type("fi_cntr_attr")
        .allowlist_type("fi_context")
        .allowlist_type("fi_context2")
        .allowlist_type("fi_cq_entry")
        .allowlist_type("fi_cq_msg_entry")
        .allowlist_type("fi_cq_data_entry")
        .allowlist_type("fi_cq_tagged_entry")
        .allowlist_type("fi_cq_err_entry")
        .allowlist_type("fi_rma_iov")
        .allowlist_type("fi_msg_rma")
        .allowlist_type("iovec")
        // Derive useful traits
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        // Size_t and other platform types
        .size_t_is_usize(true)
        .generate()
        .expect("Failed to generate libfabric bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}
