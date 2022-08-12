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

using block_sort_t = xpu::block_sort<unsigned long int, experimental::CbmStsDigi, 64, 2>;
struct GpuSortSmem {
    using sort_buf_t = typename block_sort_t::storage_t;
    sort_buf_t sortbuf;
};

XPU_KERNEL(BlockSort, GpuSortSmem, experimental::CbmStsDigi* data, experimental::CbmStsDigi* buf, experimental::CbmStsDigi** out, const size_t numElems) {
    experimental::CbmStsDigi* res = block_sort_t(smem.sortbuf).sort(data, numElems, buf, [](const experimental::CbmStsDigi& a) { return ((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time); });

    if (xpu::block_idx::x() == 0) {
        *out = res;
    }
}

// Optional shorthand for the sorting class.
//
// Template arguments are the type of the key that is sorted,
// size of the gpu block (currently hard-coded at 64 threads)
// and the number of keys that are sorted by each thread with
// the underlying cub::BlockRadixSort implementation.
/*
using SortT = xpu::block_sort<unsigned long int, experimental::CbmStsDigi, CBM_STS_SORT_BLOCK_SIZE, CBM_STS_SORT_ITEMS_PER_THREAD>;

// Define type that is used to allocate shared memory.
// In this case only shared memory for the underlying cub::BlockRadixSort is needed.
struct GpuSortSmem {
    typename SortT::storage_t sortBuf;
};

XPU_KERNEL(BlockSort, GpuSortSmem, experimental::CbmStsDigi* data, experimental::CbmStsDigi* buf, experimental::CbmStsDigi** out, const size_t numElems) {
    // Call the sort function. Along the two buffers and the number of elements, a function that
    // extracts the key from the struct has to be passed.
    // Returns the buffer that contains the sorted data (either data or buf).

    const size_t itemsPerBlock = numElems / xpu::block_dim::x();
    const size_t itemsPerThread = itemsPerBlock / 64;
    const size_t offset = xpu::thread_idx::x() * itemsPerThread + itemsPerBlock * xpu::block_idx::x();
    //printf("block_idx=%d, block_dim:x=%d, xpu::grid_dim::x=%d, xpu::thread_idx::x=%d, itemsPerBlock=%lu, itemsPerThread=%lu, offset=%lu, n=%lu\n", xpu::block_idx::x(), xpu::block_dim::x(), xpu::grid_dim::x(), xpu::thread_idx::x(), itemsPerBlock, itemsPerThread, offset, numElems);

    if ((offset + itemsPerThread) < numElems) {
    experimental::CbmStsDigi* res = SortT(smem.sortBuf).sort(
        &data[offset], itemsPerThread, &buf[offset],
        [](const experimental::CbmStsDigi& a) { return ((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time); }
    );

    if (xpu::block_idx::x() == 0) {
        *out = res;
    }
    }
}
*/
