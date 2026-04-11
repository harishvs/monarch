/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Wrapper header for bindgen — includes all libfabric headers needed
   by the OFI RDMA backend.

   Many libfabric functions are defined as `static inline` in the C
   headers, which bindgen cannot generate bindings for. Plain-C wrappers
   are implemented in wrapper.c and declared here so bindgen can see them.
   Each wrapper is prefixed with `libfabric_sys_` to avoid name collisions. */

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>

/* FI_LOCAL_MR was deprecated in libfabric 2.x (uses _Pragma, which
   bindgen cannot parse). Expose the raw value as a plain constant. */
#define LIBFABRIC_SYS_FI_LOCAL_MR (1ULL << 55)

/* -----------------------------------------------------------------------
   Declarations for plain-C wrappers (implemented in wrapper.c).
   ----------------------------------------------------------------------- */

int libfabric_sys_fi_close(struct fid *fid);

int libfabric_sys_fi_domain(
    struct fid_fabric *fabric, struct fi_info *info,
    struct fid_domain **domain, void *context);

int libfabric_sys_fi_endpoint(
    struct fid_domain *domain, struct fi_info *info,
    struct fid_ep **ep, void *context);

int libfabric_sys_fi_ep_bind(
    struct fid_ep *ep, struct fid *bfid, uint64_t flags);

int libfabric_sys_fi_enable(struct fid_ep *ep);

int libfabric_sys_fi_getname(
    fid_t fid, void *addr, size_t *addrlen);

int libfabric_sys_fi_av_open(
    struct fid_domain *domain, struct fi_av_attr *attr,
    struct fid_av **av, void *context);

int libfabric_sys_fi_av_insert(
    struct fid_av *av, const void *addr, size_t count,
    fi_addr_t *fi_addr, uint64_t flags, void *context);

int libfabric_sys_fi_cq_open(
    struct fid_domain *domain, struct fi_cq_attr *attr,
    struct fid_cq **cq, void *context);

ssize_t libfabric_sys_fi_cq_read(
    struct fid_cq *cq, void *buf, size_t count);

ssize_t libfabric_sys_fi_cq_readerr(
    struct fid_cq *cq, struct fi_cq_err_entry *buf, uint64_t flags);

const char *libfabric_sys_fi_cq_strerror(
    struct fid_cq *cq, int prov_errno, const void *err_data,
    char *buf, size_t len);

int libfabric_sys_fi_mr_reg(
    struct fid_domain *domain, const void *buf, size_t len,
    uint64_t access, uint64_t offset, uint64_t requested_key,
    uint64_t flags, struct fid_mr **mr, void *context);

int libfabric_sys_fi_mr_regattr(
    struct fid_domain *domain, const struct fi_mr_attr *attr,
    uint64_t flags, struct fid_mr **mr);

uint64_t libfabric_sys_fi_mr_key(struct fid_mr *mr);

void *libfabric_sys_fi_mr_desc(struct fid_mr *mr);

ssize_t libfabric_sys_fi_send(
    struct fid_ep *ep, const void *buf, size_t len, void *desc,
    fi_addr_t dest_addr, void *context);

ssize_t libfabric_sys_fi_recv(
    struct fid_ep *ep, void *buf, size_t len, void *desc,
    fi_addr_t src_addr, void *context);

ssize_t libfabric_sys_fi_write(
    struct fid_ep *ep, const void *buf, size_t len, void *desc,
    fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context);

ssize_t libfabric_sys_fi_read(
    struct fid_ep *ep, void *buf, size_t len, void *desc,
    fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context);
