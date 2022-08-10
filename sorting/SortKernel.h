#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"

struct BlockSortKernel {};
XPU_EXPORT_KERNEL(BlockSortKernel, BlockSort, experimental::CbmStsDigi*, experimental::CbmStsDigi*, experimental::CbmStsDigi**, size_t);

struct JanSergeySortKernel{};
XPU_EXPORT_KERNEL(JanSergeySortKernel, JanSergeySort, const int, const experimental::CbmStsDigi*, const int*, const int*, experimental::CbmStsDigi*);

