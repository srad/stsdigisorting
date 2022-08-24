#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"
#include "../constants.h"

struct JanSergeySortSingleBlockKernel{};
XPU_EXPORT_KERNEL(JanSergeySortSingleBlockKernel, JanSergeySortSingleBlock, const size_t, const digi_t*, const int*, const int*, digi_t*, const unsigned int*);
XPU_BLOCK_SIZE_1D(JanSergeySortSingleBlock,  experimental::JanSergeySortTPB);
