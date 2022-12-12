#pragma once

#include <xpu/device.h>
#include "../datastructures.h"
#include "../constants.h"
#include "../types.h"

namespace experimental {

    struct AddressBucketsKernel{};
    XPU_EXPORT_KERNEL(AddressBucketsKernel, AddressBuckets, const size_t, const CbmStsDigiInput*, index_t*, index_t*, const index_t*, size_t, unsigned short, digi_t*);

}

XPU_BLOCK_SIZE_1D(experimental::AddressBuckets,  experimental::AddressBucketsBlockDimX);