#include <xpu/device.h>
#include "Partition.h"

XPU_IMAGE(experimental::PartitionKernel);

namespace experimental {
    struct PartitionSmem {
        unsigned int pivotIdx;
    };

    XPU_KERNEL(Partition, PartitionSmem, const digi_t* input, const index_t* startIndex, const index_t* endIndex, digi_t* output, const size_t n) {
        const auto bucketIdx = xpu::block_idx::x();
        const index_t bucketStartIdx = startIndex[bucketIdx];
        const index_t bucketEndIdx = endIndex[bucketIdx];

        if (xpu::thread_idx::x() == 0) {
            smem.pivotIdx = 0;
        }
        xpu::barrier();

        // -----------------------------------------------------------------------------------------------------------
        // Phase 1. Count the output array layout of a two partitioned array, i.e.:
        //
        //                                  Pivot index (i)
        //                                      v
        // +------------------------------------+------------------------------+
        // |       # digis channels < 1024      |    # digis channels >= 1024  |
        // +------------------------------------+------------------------------+
        // |  0  |  1  |  2  |  ...         | i | i+1 | i+2 | ...        | n-1 |
        // +-----+------------------------------+------------------------------+
        //
        for (auto i = bucketStartIdx + xpu::thread_idx::x(); i <= bucketEndIdx; i += xpu::block_dim::x()) {
            xpu::atomic_add_block(&smem.pivotIdx, (input[i].channel < 1024));
        }
        xpu::barrier();

        //if (xpu::thread_idx::x() == 0) {
    }
}
