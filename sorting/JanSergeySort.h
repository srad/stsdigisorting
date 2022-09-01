#pragma once

#include <xpu/device.h>
#include <cstddef> // for size_t
#include "../datastructures.h"
#include "../constants.h"
#include "../types.h"

namespace experimental {

    constexpr count_t channelRange = channelCount / 2;
    constexpr count_t itemsPerBlock = channelRange / JanSergeySortTPB;

    static_assert(channelRange > 0, "JanSergeySort: channelRange is not positive");
    static_assert(itemsPerBlock > 0, "JanSergeySort: itemsPerBlock is not positive");

    struct JanSergeySortKernel{};
    XPU_EXPORT_KERNEL(JanSergeySortKernel, JanSergeySort, const size_t, const digi_t*, const index_t*, const index_t*, digi_t*, const index_t*);

}

XPU_BLOCK_SIZE_1D(experimental::JanSergeySort,  experimental::JanSergeySortTPB);