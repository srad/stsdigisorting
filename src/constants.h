#pragma once

namespace experimental {
    constexpr int channelCount = 2048;
    constexpr int WarpMultiplier = 3;

  #if XPU_IS_CUDA
    constexpr int WarpSize = 32;
  #else
    constexpr int WarpSize = 64;
  #endif

    constexpr int JanSergeySortBlockDimX = WarpSize * WarpMultiplier;
    constexpr int PartitionBlockDimX = WarpSize * WarpMultiplier;
    constexpr int BlockSortBlockDimX = 64;
    constexpr int BlockSortItemsPerThread = 8;
}
