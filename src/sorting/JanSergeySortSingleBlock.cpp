#include <xpu/device.h>
#include "JanSergeySortSingleBlock.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"

XPU_IMAGE(experimental::JanSergeySortSingleBlockKernel);

namespace experimental {

    constexpr unsigned int channelRange = channelCount;
    constexpr unsigned int itemsPerBlock = channelRange / JanSergeySortBlockDimX;

    using block_scan_t = xpu::block_scan<count_t, JanSergeySortBlockDimX>;

    struct JanSergeySortSingleBlockSmem {
        count_t channelOffset[channelRange];
        block_scan_t::storage_t temp;
    };

    XPU_KERNEL(JanSergeySortSingleBlock, JanSergeySortSingleBlockSmem, const size_t n, const digi_t* digis, const index_t* startIndex, const index_t* endIndex, digi_t* output) {
        const auto bucketIdx = xpu::block_idx::x();
        const index_t bucketStartIdx = startIndex[bucketIdx];
        const index_t bucketEndIdx = endIndex[bucketIdx];

        const index_t itemsPerBlockOffset = xpu::thread_idx::x() * itemsPerBlock;

        // -----------------------------------------------------------------------------------------------------------
        // Phase 1. Init all channel counters to zero: O(channelCount) = O(1)
        // -----------------------------------------------------------------------------------------------------------
        for (auto i = xpu::thread_idx::x(); i < channelCount; i += xpu::block_dim::x()) {
            smem.channelOffset[i] = 0;
        }
        xpu::barrier();

        // -----------------------------------------------------------------------------------------------------------
        // Phase 2. Count channels: O(n/p)
        // -----------------------------------------------------------------------------------------------------------       
        for (auto i = bucketStartIdx + xpu::thread_idx::x(); i <= bucketEndIdx; i += xpu::block_dim::x()) {
            xpu::atomic_add_block(&smem.channelOffset[digis[i].channel], 1);
        }
        xpu::barrier();

        /*
        //
        // In this code base there is a bug in the CUB code below for exclusive-sum.
        // However, the loop below is hardly slower (O(1) anyways) and CPU compatible.
        //
        // -----------------------------------------------------------------------------------------------------------
        // Phase 3. Exclusive sum: O(channelCount) = O(1)
        // -----------------------------------------------------------------------------------------------------------
        block_scan_t scan{smem.temp};
        const auto channelStartIndex = xpu::thread_idx::x() * itemsPerBlock;

        count_t items[itemsPerBlock];
        for(int i=0; (i < itemsPerBlock) && ((channelStartIndex + i) < channelCount); i++) {
            items[i] = smem.channelOffset[channelStartIndex + i];
        }
        
        // Collectively compute the block-wide inclusive prefix sum
        // channelOffset + offset static_cast, mal ausprobieren
        scan.exclusive_sum(items, items);
        xpu::barrier();

        for(int i=0; (i < itemsPerBlock) && ((channelStartIndex + i) < channelCount); i++) {
            smem.channelOffset[channelStartIndex + i] = items[i];
        }
        xpu::barrier();
        */

        // -----------------------------------------------------------------------------------------------------------
        // Phase 4. Final sorting, place the elements in the correct position within the global output array: O(n)
        //
        // This must be done linearly otherwise there will be race conditions if a thread on the right wants to insert
        // a digis into the same channel as a thread on the left. Might be handled by some algoritm, unclear yet.
        // -----------------------------------------------------------------------------------------------------------
        if (xpu::thread_idx::x() == 0) {
            // -----------------------------------------------------------------------------------------------------------
            // 3. Exclusive sum: O(channelCount) -> O(1)
            // -----------------------------------------------------------------------------------------------------------
            count_t sum = 0;
            for (int i = 0; i < channelCount; i++) {
                const auto tmp = smem.channelOffset[i];
                smem.channelOffset[i] = sum;
                sum += tmp;
            }

            for (auto i = bucketStartIdx; i <= bucketEndIdx; i++) {
                output[bucketStartIdx + (smem.channelOffset[digis[i].channel]++)] = digis[i];
            }
        }
    }
}