#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"

struct BlockSortKernel {};
XPU_EXPORT_KERNEL(BlockSortKernel, BlockSort, digi_t*, const int*, const int*, digi_t*, digi_t**, const size_t);
