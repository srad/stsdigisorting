#include <xpu/device.h>
#include "JanSergeySortSimpleSum.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"

XPU_IMAGE(experimental::JanSergeySortSimpleSumKernel);

namespace experimental {

    struct JanSergeySortSimpleSumSmem {
        count_t channelOffset[channelCount];
    };

    XPU_KERNEL(JanSergeySortSimpleSum, JanSergeySortSimpleSumSmem, const size_t n, const digi_t* digis, const index_t* startIndex, const index_t* endIndex, digi_t* output, const index_t* channelSplitIndex) {
        const index_t bucketIdx = xpu::block_idx::x();
        const index_t bucketStartIdx = startIndex[bucketIdx];
        const index_t bucketEndIdx = endIndex[bucketIdx];

        // -----------------------------------------------------------------------------------------------------------
        // 1. Init all channel counters to zero: O(channelCount) -> O(1)
        // This step is not related to the input size.
        // -----------------------------------------------------------------------------------------------------------
        for (index_t i = xpu::thread_idx::x(); i < channelCount; i += xpu::block_dim::x()) {
            smem.channelOffset[i] = 0;
        }
        xpu::barrier();

        // -----------------------------------------------------------------------------------------------------------
        // 2. Count channels: O(n + channelCount) -> O(n)
        // -----------------------------------------------------------------------------------------------------------
        const index_t threadStart = bucketStartIdx + xpu::thread_idx::x();

        for (index_t i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
            xpu::atomic_add_block(&smem.channelOffset[digis[i].channel], 1);
        }
        xpu::barrier();   

        if (xpu::thread_idx::x() == 0) {
            // -----------------------------------------------------------------------------------------------------------
            // 3. Exclusive sum: O(channelCount) -> O(1)
            // -----------------------------------------------------------------------------------------------------------
            unsigned int sum = 0;
            for (int i = 0; i < channelCount; i++) {
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
            for (index_t i = bucketStartIdx; i <= bucketEndIdx; i++) {
                output[bucketStartIdx + (smem.channelOffset[digis[i].channel]++)] = digis[i];
            }
        }
    }
}