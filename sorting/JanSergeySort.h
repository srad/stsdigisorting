#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"

struct JanSergeySortKernel{};
XPU_EXPORT_KERNEL(JanSergeySortKernel, JanSergeySort, const size_t, const experimental::CbmStsDigi*, const int*, const int*, experimental::CbmStsDigi*);
XPU_BLOCK_SIZE(JanSergeySort, 1024);
