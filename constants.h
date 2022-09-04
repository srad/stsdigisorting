
#pragma once

#define DEBUG_SORT

namespace experimental {
    constexpr int channelCount = 2048;
    constexpr int JanSergeySortTPB = 128;
    constexpr int BlockSortBlockDimX = 64;
    constexpr int BlockSortItemsPerThread = 8;
}

