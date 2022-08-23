#pragma once

namespace experimental {

#if XPU_IS_CUDA
    constexpr int channelCount = 2048;
    constexpr int JanSergeySortTPB = 128;

    constexpr int BlockSortBlockDimX = 64;
    constexpr int BlockSortItemsPerThread = 8;
#else
    constexpr int channelCount = 2048;
    constexpr int JanSergeySortTPB = 1024;

    constexpr int BlockSortBlockDimX = 64;
    constexpr int BlockSortItemsPerThread = 8;
#endif

}