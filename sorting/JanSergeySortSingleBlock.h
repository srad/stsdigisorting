#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"
#include "../constants.h"
#include "../types.h"

namespace experimental {

    struct JanSergeySortSingleBlockKernel{};
    XPU_EXPORT_KERNEL(JanSergeySortSingleBlockKernel, JanSergeySortSingleBlock, const size_t, const digi_t*, const index_t*, const index_t*, digi_t*, const index_t*);

}

XPU_BLOCK_SIZE_1D(experimental::JanSergeySortSingleBlock, experimental::JanSergeySortBlockDimX);
