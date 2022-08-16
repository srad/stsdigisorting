#include <xpu/device.h>
#include "SortKernel.h"
#include "../datastructures.h"
#include "../common.h"

// Initialize the xpu image. This macro must be placed once somewhere in the device sources.
XPU_IMAGE(BlockSortKernel);

#if XPU_IS_CUDA
#define CBM_STS_SORT_BLOCK_SIZE 32
#define CBM_STS_SORT_ITEMS_PER_THREAD 32
#else
#define CBM_STS_SORT_BLOCK_SIZE 256
#define CBM_STS_SORT_ITEMS_PER_THREAD 6
#endif

// <Bock size=thread per block (hier zumindest), items per thread>
using SortT = xpu::block_sort<unsigned long int, experimental::CbmStsDigi, 64, 2>;

struct GpuSortSmem {
    typename SortT::storage_t sortBuf;
};

XPU_KERNEL(BlockSort, GpuSortSmem, experimental::CbmStsDigi* data, int* startIndex, int* endIdex, experimental::CbmStsDigi* buf, experimental::CbmStsDigi** out, const size_t numElems) {
    const size_t itemsPerThread = endIdex[xpu::block_idx::x()] - startIndex[xpu::block_idx::x()] + 1;
    const size_t offset = startIndex[xpu::block_idx::x()];

    // Do not overshoot array boundary when you have more blocks than threads.
    experimental::CbmStsDigi* res = SortT(smem.sortBuf).sort(
        &data[offset], itemsPerThread, &buf[offset],
        [](const experimental::CbmStsDigi& a) { return ((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time); }
    );

    // Once the sorting is completed, the first thread in each Block write
    if (xpu::thread_idx::x() == 0) {
        out[xpu::block_idx::x()] = res;
    }
}