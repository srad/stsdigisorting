#include <xpu/device.h>
#include "JanSergeySortSingleBlock.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"

XPU_IMAGE(JanSergeySortSingleBlockKernel);

constexpr unsigned int channelRange = experimental::channelCount;
constexpr unsigned int itemsPerBlock = channelRange / experimental::JanSergeySortTPB;

using count_t = unsigned int;
using block_scan_t = xpu::block_scan<unsigned int, channelRange>;

struct JanSergeySortSingleBlockSmem {
    count_t channelOffset[channelRange];
    block_scan_t::storage_t temp;
};

// 
// This kernel ignores the argument: channelSplitIndex
// 
XPU_KERNEL(JanSergeySortSingleBlock, JanSergeySortSingleBlockSmem, const size_t n, const digi_t* digis, const int* startIndex, const int* endIndex, digi_t* output, const unsigned int* channelSplitIndex) {
    const int bucketIdx = xpu::block_idx::x();
    const int bucketStartIdx = startIndex[bucketIdx];
    const int bucketEndIdx = endIndex[bucketIdx];
    const int threadStart = bucketStartIdx + xpu::thread_idx::x();

    // -----------------------------------------------------------------------------------------------------------
    // 1. Init all channel counters to zero: O(channelCount)
    // This step is not related to the input size.
    // -----------------------------------------------------------------------------------------------------------
    for (int i = xpu::thread_idx::x(); i < channelRange; i += xpu::block_dim::x()) {
        smem.channelOffset[i] = 0;
    }
    xpu::barrier();

    // -----------------------------------------------------------------------------------------------------------
    // 2. Count channels: O(n + channelCount) -> O(n)
    // -----------------------------------------------------------------------------------------------------------
    for (int i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        xpu::atomic_add_block(&smem.channelOffset[digis[i].channel], 1);
    }
    xpu::barrier();

    if (xpu::thread_idx::x() == 0) {
        // -----------------------------------------------------------------------------------------------------------
        // 3. Exclusive sum: O(channelCount)
        // -----------------------------------------------------------------------------------------------------------
        count_t sum = 0;
        for (int i = 0; i < experimental::channelCount; i++) {
            const auto tmp = smem.channelOffset[i];
            smem.channelOffset[i] = sum;
            sum += tmp;
        }
        // -----------------------------------------------------------------------------------------------------------
        // 4. Final sorting, place the elements in the correct position within the global output array: O(n)
        //
        // This must be done linearly otherwise there will be race conditions if a thread on the right wants to insert
        // a digis into the same channel as a thread on the left. Might be handled by some algoritm, unclear yet.
        // -----------------------------------------------------------------------------------------------------------

        for (int i = bucketStartIdx; i <= bucketEndIdx; i++) {
            output[bucketStartIdx + (smem.channelOffset[digis[i].channel]++)] = digis[i];
        }
    }
}