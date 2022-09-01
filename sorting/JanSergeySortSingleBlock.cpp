#include <xpu/device.h>
#include "JanSergeySortSingleBlock.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"

XPU_IMAGE(experimental::JanSergeySortSingleBlockKernel);

namespace experimental {

    constexpr unsigned int channelRange = channelCount;
    constexpr unsigned int itemsPerBlock = channelRange / JanSergeySortTPB;

    static_assert(channelRange > 0, "JanSergeySortSingleBlockKernel: channelRange is not positive");
    static_assert(itemsPerBlock > 0, "JanSergeySortSingleBlockKernel: itemsPerBlock is not positive");

    using block_scan_t = xpu::block_scan<count_t, JanSergeySortTPB>;

    struct JanSergeySortSingleBlockSmem {
        count_t channelOffset[channelRange];
        // Each thread must know how many threads came to before it.
        // unsigned short channelCountPerItemBlock[JanSergeySortTPB][channelRange];
        block_scan_t::storage_t temp;
    };

    XPU_KERNEL(JanSergeySortSingleBlock, JanSergeySortSingleBlockSmem, const size_t n, const digi_t* digis, const index_t* startIndex, const index_t* endIndex, digi_t* output, const index_t* channelSplitIndex) {
        const index_t bucketIdx = xpu::block_idx::x();
        const index_t bucketStartIdx = startIndex[bucketIdx];
        const index_t bucketEndIdx = endIndex[bucketIdx];

        const index_t threadStart = bucketStartIdx + xpu::thread_idx::x();

        const index_t itemsPerBlockOffset = xpu::thread_idx::x() * itemsPerBlock;

        // -----------------------------------------------------------------------------------------------------------
        // 1. Init all channel counters to zero: O(channelCount)
        // This step is not related to the input size.
        // -----------------------------------------------------------------------------------------------------------
        for (index_t i = xpu::thread_idx::x(); i < channelRange; i += xpu::block_dim::x()) {
            smem.channelOffset[i] = 0;
            //smem.channelCountPerItemBlock[xpu::thread_idx::x()][i] = 0;
        }
        xpu::barrier();

        // -----------------------------------------------------------------------------------------------------------
        // 2. Count channels: O(n + channelCount) -> O(n)
        // -----------------------------------------------------------------------------------------------------------
        for (index_t i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
            xpu::atomic_add_block(&smem.channelOffset[digis[i].channel], 1);
        }
        xpu::barrier();   

        // -----------------------------------------------------------------------------------------------------------
        // 3. Exclusive sum: O(channelCount)
        // -----------------------------------------------------------------------------------------------------------
        block_scan_t scan{smem.temp};

        const uint_t channelStartIndex = xpu::thread_idx::x() * itemsPerBlock;
        auto channelOffsetSection = reinterpret_cast<count_t(*)[itemsPerBlock]>(smem.channelOffset + channelStartIndex);
        scan.exclusive_sum(*channelOffsetSection, *channelOffsetSection);
        xpu::barrier();

        if (xpu::thread_idx::x() == 0) {
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