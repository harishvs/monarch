/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Plain-C wrappers for static-inline libfabric functions.
 *
 * Bindgen cannot generate Rust FFI bindings for `static inline`
 * functions.  This file provides non-inline wrappers that are compiled
 * into a small static library and linked into the final binary. */

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>

/* --- Core lifecycle ---------------------------------------------------- */

int libfabric_sys_fi_close(struct fid *fid)
{
    return fi_close(fid);
}

int libfabric_sys_fi_domain(
    struct fid_fabric *fabric, struct fi_info *info,
    struct fid_domain **domain, void *context)
{
    return fi_domain(fabric, info, domain, context);
}

/* --- Endpoint ---------------------------------------------------------- */

int libfabric_sys_fi_endpoint(
    struct fid_domain *domain, struct fi_info *info,
    struct fid_ep **ep, void *context)
{
    return fi_endpoint(domain, info, ep, context);
}

int libfabric_sys_fi_ep_bind(
    struct fid_ep *ep, struct fid *bfid, uint64_t flags)
{
    return fi_ep_bind(ep, bfid, flags);
}

int libfabric_sys_fi_enable(struct fid_ep *ep)
{
    return fi_enable(ep);
}

int libfabric_sys_fi_getname(
    fid_t fid, void *addr, size_t *addrlen)
{
    return fi_getname(fid, addr, addrlen);
}

/* --- Address vector ---------------------------------------------------- */

int libfabric_sys_fi_av_open(
    struct fid_domain *domain, struct fi_av_attr *attr,
    struct fid_av **av, void *context)
{
    return fi_av_open(domain, attr, av, context);
}

int libfabric_sys_fi_av_insert(
    struct fid_av *av, const void *addr, size_t count,
    fi_addr_t *fi_addr, uint64_t flags, void *context)
{
    return fi_av_insert(av, addr, count, fi_addr, flags, context);
}

/* --- Completion queue -------------------------------------------------- */

int libfabric_sys_fi_cq_open(
    struct fid_domain *domain, struct fi_cq_attr *attr,
    struct fid_cq **cq, void *context)
{
    return fi_cq_open(domain, attr, cq, context);
}

ssize_t libfabric_sys_fi_cq_read(
    struct fid_cq *cq, void *buf, size_t count)
{
    return fi_cq_read(cq, buf, count);
}

ssize_t libfabric_sys_fi_cq_readerr(
    struct fid_cq *cq, struct fi_cq_err_entry *buf, uint64_t flags)
{
    return fi_cq_readerr(cq, buf, flags);
}

const char *libfabric_sys_fi_cq_strerror(
    struct fid_cq *cq, int prov_errno, const void *err_data,
    char *buf, size_t len)
{
    return fi_cq_strerror(cq, prov_errno, err_data, buf, len);
}

/* --- Memory registration ----------------------------------------------- */

int libfabric_sys_fi_mr_reg(
    struct fid_domain *domain, const void *buf, size_t len,
    uint64_t access, uint64_t offset, uint64_t requested_key,
    uint64_t flags, struct fid_mr **mr, void *context)
{
    return fi_mr_reg(domain, buf, len, access, offset, requested_key,
                     flags, mr, context);
}

int libfabric_sys_fi_mr_regattr(
    struct fid_domain *domain, const struct fi_mr_attr *attr,
    uint64_t flags, struct fid_mr **mr)
{
    return domain->mr->regattr(&domain->fid, attr, flags, mr);
}

uint64_t libfabric_sys_fi_mr_key(struct fid_mr *mr)
{
    return fi_mr_key(mr);
}

void *libfabric_sys_fi_mr_desc(struct fid_mr *mr)
{
    return fi_mr_desc(mr);
}

/* --- RMA (RDMA read/write) --------------------------------------------- */

ssize_t libfabric_sys_fi_write(
    struct fid_ep *ep, const void *buf, size_t len, void *desc,
    fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context)
{
    return fi_write(ep, buf, len, desc, dest_addr, addr, key, context);
}

ssize_t libfabric_sys_fi_read(
    struct fid_ep *ep, void *buf, size_t len, void *desc,
    fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context)
{
    return fi_read(ep, buf, len, desc, src_addr, addr, key, context);
}
