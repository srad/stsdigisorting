#include <xpu/device.h>
#include "BlockSort.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"
#include "../constants.h"

// Initialize the xpu image. This macro must be placed once somewhere in the device sources.
XPU_IMAGE(BlockSortKernel);

// <Bock size=thread per block (hier zumindest), items per thread>
using SortT = xpu::block_sort<unsigned long int, experimental::CbmStsDigi, experimental::BlockSortBlockDimX, experimental::BlockSortItemsPerThread>;

struct GpuSortSmem {
    typename SortT::storage_t sortBuf;
};

XPU_KERNEL(BlockSort, GpuSortSmem, experimental::CbmStsDigi* data, int* startIndex, int* endIndex, experimental::CbmStsDigi* buf, experimental::CbmStsDigi** out, const size_t numElems, unsigned int* sideStartIndex, unsigned int* sideEndIndex) {
    const size_t itemsPerThread = endIndex[xpu::block_idx::x()] - startIndex[xpu::block_idx::x()] + 1;
    const size_t offset = startIndex[xpu::block_idx::x()];
    const int bucketIdx = xpu::block_idx::x();
    const int bucketStartIdx = startIndex[bucketIdx];
    const int bucketEndIdx = endIndex[bucketIdx];

    // Do not overshoot array boundary when you have more blocks than threads.
    experimental::CbmStsDigi* res = SortT(smem.sortBuf).sort(
        &data[offset], itemsPerThread, &buf[offset],
        [](const experimental::CbmStsDigi& a) { return ((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time); }
    );

    // Once the sorting is completed, the first thread in each Block write
    if (xpu::thread_idx::x() == 0) {
        //printf("offset=%d, itemsPerThread=%d, xpu::block_idx::x()=%d, side_index=%d\n", offset, itemsPerThread, xpu::block_idx::x(), findSideSeperatorIndex(res, itemsPerThread, sideSeperator));
        const int index = findSideSeperatorIndex(res, itemsPerThread, sideSeperator);
        
        sideStartIndex[bucketIdx * 2] = bucketStartIdx;
        sideEndIndex[bucketIdx * 2] =  bucketStartIdx + index;

        // Back
        sideStartIndex[bucketIdx * 2 + 1] = sideEndIndex[bucketIdx * 2] + 1;
        sideEndIndex[bucketIdx * 2 + 1] = bucketEndIdx;

        //printf("offset=%d, itemsPerThread=%d, xpu::block_idx::x()=%d, side_index=%d\n", offset, itemsPerThread, xpu::block_idx::x(), findSideSeperatorIndex(res, itemsPerThread, sideSeperator));
        out[xpu::block_idx::x()] = res;
    }
}