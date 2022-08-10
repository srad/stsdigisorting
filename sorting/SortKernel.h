#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"

struct BlockSortKernel {};
XPU_EXPORT_KERNEL(BlockSortKernel, BlockSort, experimental::CbmStsDigi*, experimental::CbmStsDigi*, experimental::CbmStsDigi**, size_t);
