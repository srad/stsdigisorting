#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"

struct JanSergeySortSidedKernel{};
XPU_EXPORT_KERNEL(JanSergeySortSidedKernel, JanSergeySortSided, const size_t, const experimental::CbmStsDigi*, const unsigned int*, const unsigned int*, experimental::CbmStsDigi*, unsigned int*, unsigned int*);
XPU_BLOCK_SIZE(JanSergeySortSided, 1024);
