#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"
#include "../types.h"

namespace experimental {

    struct BlockSortKernel {};
    XPU_EXPORT_KERNEL(BlockSortKernel, BlockSort, digi_t*, const index_t*, const index_t*, digi_t*, digi_t**, const size_t);

}

XPU_BLOCK_SIZE_1D(unsigned int, experimental::BlockSortBlockDimX);