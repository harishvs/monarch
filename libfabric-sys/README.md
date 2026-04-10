# libfabric-sys

Raw FFI bindings to [libfabric](https://ofiwg.github.io/libfabric/) (OpenFabrics Interfaces).

## Why wrapper.c exists

Most libfabric functions (`fi_close`, `fi_write`, `fi_read`, `fi_mr_reg`, etc.) are defined as `static inline` in C headers. Bindgen can't generate Rust bindings for these. `wrapper.c` provides plain-C wrapper functions with external linkage that call the real ones. These are compiled into a static library via the `cc` crate and linked into the final binary.

## Build

The build script (`build.rs`):
1. Finds libfabric via pkg-config or common install paths (`/opt/amazon/efa`, `/usr/local`, `/usr`)
2. Compiles `wrapper.c` into `libfabric_wrapper.a`
3. Generates Rust bindings from `wrapper.h` using bindgen

## Safe helpers

`lib.rs` provides cached availability probes:
- `ofi_available()` — any usable libfabric provider present
- `efa_ofi_available()` — EFA provider specifically
- `version()` — libfabric API version
