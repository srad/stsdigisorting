#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"
#include "../constants.h"

struct JanSergeySortKernel{};
XPU_EXPORT_KERNEL(JanSergeySortKernel, JanSergeySort, const size_t, const digi_t*, const int*, const int*, digi_t*, const unsigned int*);
XPU_BLOCK_SIZE_1D(JanSergeySort,  experimental::JanSergeySortTPB);
