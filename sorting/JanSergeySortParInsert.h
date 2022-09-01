#pragma once

#include <xpu/device.h>
#include "../datastructures.h"
#include "../constants.h"
#include "../types.h"

namespace experimental {

    struct JanSergeySortParInsertKernel{};
    XPU_EXPORT_KERNEL(JanSergeySortParInsertKernel, JanSergeySortParInsert, const size_t, const digi_t*, const index_t*, const index_t*, digi_t*, const index_t*);

}

XPU_BLOCK_SIZE_1D(experimental::JanSergeySortParInsert,  experimental::JanSergeySortTPB);