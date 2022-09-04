#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"
#include "../constants.h"
#include "../types.h"

namespace experimental {

    struct JanSergeySortSimpleSumKernel{};
    XPU_EXPORT_KERNEL(JanSergeySortSimpleSumKernel, JanSergeySortSimpleSum, const size_t, const digi_t*, const index_t*, const index_t*, digi_t*, const index_t*);

}

XPU_BLOCK_SIZE_1D(experimental::JanSergeySortSimpleSum, experimental::JanSergeySortTPB);
