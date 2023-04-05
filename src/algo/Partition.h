#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"
#include "../constants.h"
#include "../types.h"

namespace experimental {

    struct PartitionKernel {};
    XPU_EXPORT_KERNEL(PartitionKernel, Partition, const digi_t*, const index_t*, const index_t*, digi_t*, const size_t n);

}

XPU_BLOCK_SIZE_1D(experimental::Partition, experimental::PartitionBlockDimX);