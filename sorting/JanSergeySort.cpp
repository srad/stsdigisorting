#include <xpu/device.h>
#include "JanSergeySort.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"

XPU_IMAGE(JanSergeySortKernel);

//constexpr int block_scan_block_dim_x = 64;
constexpr unsigned int channelRange = experimental::channelCount / 2;
constexpr unsigned int itemsPerBlock = channelRange / experimental::JanSergeySortTPB;

using count_t = unsigned int;
using block_scan_t = xpu::block_scan<unsigned int, channelRange>;

struct JanSergeySortSmem {
    count_t channelOffset[channelRange];
    block_scan_t::storage_t temp;
};

XPU_KERNEL(JanSergeySort, JanSergeySortSmem, const size_t n, const experimental::CbmStsDigi* digis, const int* startIndex, const int* endIndex, experimental::CbmStsDigi* output, const unsigned int* channelSplitIndex) {

    // +--------------------------------------------------------------------+
    // | Bucket 0             | Bucket 1             | Bucket 2             |
    // +---------+------------+---------+------------+---------+------------+
    // | Block 0 | Block 1    | Block 2 | Block 3    | Block 4 | Block 5    |
    // +---------+------------+---------+------------+---------+------------+
    // | 0..1023 | 1024..2047 | 0..1023 | 1024..2047 | 0..1023 | 1024..2047 |
    // +---------+------------+---------+------------+---------+------------+
    // | Front   | Back       |  Front  | Back       | Front   | Back       |
    // +---------+------------+---------+------------+---------+------------+

    // Two blocks handle one bucket.
    // 0, 0, 1, 1, 2, 2, ...
    const int bucketIdx = xpu::block_idx::x() / 2;

    // 0, 1, 0, 1, ...
    const unsigned int blockSide = (xpu::block_idx::x() % 2) * 1;

    const bool isFront = blockSide == 0;
    const bool isBack = blockSide == 1;

    const int bucketStartIdx = (isFront * startIndex[bucketIdx]) + (isBack * channelSplitIndex[bucketIdx]);
    const int bucketEndIdx = (isFront * (channelSplitIndex[bucketIdx] - 1)) + (isBack * endIndex[bucketIdx]);
    const int threadStart = bucketStartIdx + xpu::thread_idx::x();

    // -----------------------------------------------------------------------------------------------------------
    // 1. Init all channel counters to zero: O(channelCount)
    // This step is not related to the input size, so aktually O(1).
    // -----------------------------------------------------------------------------------------------------------
    for (int i = xpu::thread_idx::x(); i < channelRange; i += xpu::block_dim::x()) {
        smem.channelOffset[i] = 0;
    }
    xpu::barrier();

    // -----------------------------------------------------------------------------------------------------------
    // 2. Count channels: O(n)
    // -----------------------------------------------------------------------------------------------------------
    for (int i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        xpu::atomic_add_block(&smem.channelOffset[digis[i].channel % 1024], 1);
    }
    xpu::barrier();

    // -----------------------------------------------------------------------------------------------------------
    // 3. Exclusive sum: O(channelCount)
    // -----------------------------------------------------------------------------------------------------------
    block_scan_t scan{smem.temp};

    const unsigned int channelStartIndex = xpu::thread_idx::x() * itemsPerBlock;

    count_t items[itemsPerBlock];
    for(int i=0; i < itemsPerBlock; i++) {
        items[i] = smem.channelOffset[channelStartIndex + i];
    }
    
    // Collectively compute the block-wide inclusive prefix sum
    scan.exclusive_sum(items, items);
    xpu::barrier();

    for(int i=0; i < itemsPerBlock; i++) {
        smem.channelOffset[channelStartIndex + i] = items[i];
    }    
    xpu::barrier();

    if (xpu::thread_idx::x() == 0) {   
        for (int i = bucketStartIdx; i <= bucketEndIdx; i++) {
            output[bucketStartIdx + (smem.channelOffset[digis[i].channel % 1024]++)] = digis[i];
        }
    }
}