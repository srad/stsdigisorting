#include <xpu/device.h>
#include "JanSergeySortParInsert.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"

XPU_IMAGE(experimental::JanSergeySortParInsertKernel);

namespace experimental {

    constexpr unsigned int channelRange = channelCount;
    constexpr unsigned int itemsPerBlock = channelRange / JanSergeySortTPB;

    struct JanSergeySortParInsertSmem {
        count_t channelOffset[channelRange];
    };

    XPU_KERNEL(JanSergeySortParInsert, JanSergeySortParInsertSmem, const size_t n, const digi_t* digis, const index_t* startIndex, const index_t* endIndex, digi_t* output, const index_t* channelSplitIndex) {
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
        const uint_t bucketIdx = xpu::block_idx::x() / 2;

        // Block side: 0, 1, 0, 1, ...
        const bool isFront = (xpu::block_idx::x() % 2) == 0;
        const bool isBack = (xpu::block_idx::x() % 2) == 1;

        // Multiply instead of "if".
        const uint_t bucketStartIdx = (isFront * startIndex[bucketIdx]) + (isBack * channelSplitIndex[bucketIdx]);
        const uint_t bucketEndIdx = (isFront * (channelSplitIndex[bucketIdx] - 1)) + (isBack * endIndex[bucketIdx]);
        const uint_t threadStart = bucketStartIdx + xpu::thread_idx::x();

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

        if (xpu::thread_idx::x() == 0) {   
            // -----------------------------------------------------------------------------------------------------------
            // 3. Exclusive sum: O(channelCount)
            // -----------------------------------------------------------------------------------------------------------
            unsigned int sum = 0;
            for (int i = 0; i < channelCount; i++) {
                const auto tmp = smem.channelOffset[i];
                smem.channelOffset[i] = sum;
                sum += tmp;
            }

            for (int i = bucketStartIdx; i <= bucketEndIdx; i++) {
                output[bucketStartIdx + (smem.channelOffset[digis[i].channel % 1024]++)] = digis[i];
            }
        }
    }
}