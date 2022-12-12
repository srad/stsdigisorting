#include <xpu/device.h>
#include "JanSergeySort.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"
#include "../types.h"

XPU_IMAGE(experimental::JanSergeySortKernel);

namespace experimental {

    // TODO: checken
    using block_scan_t = xpu::block_scan<count_t, JanSergeySortBlockDimX>;

    // union
    struct JanSergeySortSmem {
        count_t channelOffset[channelRange];
        block_scan_t::storage_t temp;
    };

    XPU_KERNEL(JanSergeySort, JanSergeySortSmem, const size_t n, const digi_t* digis, const index_t* startIndex, const index_t* endIndex, digi_t* output, const index_t* channelSplitIndex) {

        // +--------------------------------------------------------------------+
        // | Bucket 0             | Bucket 1             | Bucket 2             |
        // +---------+------------+---------+------------+---------+------------+
        // | Block 0 | Block 1    | Block 2 | Block 3    | Block 4 | Block 5    |
        // +---------+------------+---------+------------+---------+------------+
        // | 0..1023 | 1024..2047 | 0..1023 | 1024..2047 | 0..1023 | 1024..2047 |
        // +---------+------------+---------+------------+---------+------------+
        // | Front   | Back       |  Front  | Back       | Front   | Back       |
        // +---------+------------+---------+------------+---------+------------+
        //     |           |           |          |           |          |
        //     v           v           v          v           v          v            Parallel writes to output
        // +---------+------------+---------+------------+---------+------------+
        // |         |            |         |            |         |            |     Sorted blocks
        // +---------+------------+---------+------------+---------+------------+

        // Two blocks handle one bucket.
        // 0, 0, 1, 1, 2, 2, ...
        const index_t bucketIdx = xpu::block_idx::x() / 2;

        // Block side: 0, 1, 0, 1, ...
        const bool isFront = (xpu::block_idx::x() % 2) == 0;
        const bool isBack = (xpu::block_idx::x() % 2) == 1;

        // Branch free index computation.
        const index_t bucketStartIdx = (isFront * startIndex[bucketIdx]) + (isBack * channelSplitIndex[bucketIdx]);
        const index_t bucketEndIdx = (isFront * (channelSplitIndex[bucketIdx] - 1)) + (isBack * endIndex[bucketIdx]);
        const index_t threadStart = bucketStartIdx + xpu::thread_idx::x();

        // -----------------------------------------------------------------------------------------------------------
        // 1. Init all channel counters to zero: O(channelCount)
        // This step is not related to the input size, so actually runtime of: O(1)
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
        // vergleiche mit simple loop (sum bla bla)
        block_scan_t scan{smem.temp};

        const uint_t channelStartIndex = xpu::thread_idx::x() * itemsPerBlock;
        auto channelOffsetSection = reinterpret_cast<count_t(*)[itemsPerBlock]>(smem.channelOffset + channelStartIndex);
        //uint_t items[itemsPerBlock];
        //for(int i=0; i < itemsPerBlock; i++) {
        //    items[i] = smem.channelOffset[channelStartIndex + i];
        //}
        
        // Collectively compute the block-wide inclusive prefix sum
        // channelOffset + offset static_cast, mal ausprobieren
        scan.exclusive_sum(*channelOffsetSection, *channelOffsetSection);
        xpu::barrier();

        //for(int i=0; i < itemsPerBlock; i++) {
        //    smem.channelOffset[channelStartIndex + i] = items[i];
        //}
        //xpu::barrier();

        if (xpu::thread_idx::x() == 0) {   
            for (int i = bucketStartIdx; i <= bucketEndIdx; i++) {
                output[bucketStartIdx + (smem.channelOffset[digis[i].channel % 1024]++)] = digis[i];
            }
        }
    }
}