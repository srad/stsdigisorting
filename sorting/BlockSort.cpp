#include <xpu/device.h>
#include "BlockSort.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"
#include "../constants.h"

// Initialize the xpu image. This macro must be placed once somewhere in the device sources.
XPU_IMAGE(experimental::BlockSortKernel);

namespace experimental {
    // <Bock size=thread per block (hier zumindest), items per thread>
    // Items per thread nochmal, 
    using SortT = xpu::block_sort<unsigned long int, digi_t, BlockSortBlockDimX, BlockSortItemsPerThread>;

    struct GpuSortSmem {
        typename SortT::storage_t sortBuf;
    };

    XPU_KERNEL(BlockSort, GpuSortSmem, digi_t* data, const index_t* startIndex, const index_t* endIndex, digi_t* buf, digi_t** output, const size_t n) {
        const auto bucketIdx = xpu::block_idx::x();
        const auto bucketSize = endIndex[bucketIdx] - startIndex[bucketIdx] + 1;
        const auto offsetIdx = startIndex[bucketIdx];

        // Do not overshoot array boundary when you have more blocks than threads.
        digi_t* res = SortT(smem.sortBuf).sort(
            &data[offsetIdx], bucketSize, &buf[offsetIdx],
            [](const digi_t& a) { return ((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time); }
        );

        // Once the sorting is completed, the first thread in each Block write
        if (xpu::thread_idx::x() == 0) {
            //printf("offsetIdx=%d, bucketSize=%d, bucketIdx=%d, side_index=%d\n", offsetIdx, bucketSize, bucketIdx, findSideSeperatorIndex(res, bucketSize, sideSeperator));
            output[bucketIdx] = res;
        }
    }
}