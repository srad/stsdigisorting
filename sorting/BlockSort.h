#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"

struct BlockSortKernel {};
XPU_EXPORT_KERNEL(BlockSortKernel, BlockSort, experimental::CbmStsDigi*, int*, int*, experimental::CbmStsDigi*, experimental::CbmStsDigi**, size_t, unsigned int*, unsigned int*);
