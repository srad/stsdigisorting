#pragma once

namespace experimental {
    constexpr int channelCount = 2048;

    constexpr int WarpMultiplier = 16;

#if XPU_IS_CUDA
    constexpr int WarpSize = 32;
#else
    constexpr int WarpSize = 64;
#endif

    constexpr int JanSergeySortBlockDimX = WarpSize * WarpMultiplier;
    constexpr int BlockSortBlockDimX = 64;
    constexpr int BlockSortItemsPerThread = 8;

    //static_assert((JanSergeySortBlockDimX % WarpSize) == 0, Block dim X is not multiple of the warp size);
    //static_assert((BlockSortBlockDimX % WarpSize) == 0, Block dim X is not multiple of the warp size);
}
