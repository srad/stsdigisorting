#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"
#include "../constants.h"
#include "../types.h"

namespace experimental {

    struct JanSergeySortSimpleKernel{};
    XPU_EXPORT_KERNEL(JanSergeySortSimpleKernel, JanSergeySortSimple, const size_t, const digi_t*, const index_t*, const index_t*, digi_t*, const index_t*);

}

XPU_BLOCK_SIZE_1D(experimental::JanSergeySortSimple, experimental::JanSergeySortBlockDimX);
