/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Wrapper header for bindgen — includes all libfabric headers needed
   by the OFI RDMA backend. */

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
