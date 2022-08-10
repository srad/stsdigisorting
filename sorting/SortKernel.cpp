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

__device__ void prescan(int* countAndPrefixes);

// Optional shorthand for the sorting class.
//
// Template arguments are the type of the key that is sorted,
// size of the gpu block (currently hard-coded at 64 threads)
// and the number of keys that are sorted by each thread with
// the underlying cub::BlockRadixSort implementation.
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

    const size_t itemsPerBlock = numElems / xpu::grid_dim::x();
    const size_t offset = itemsPerBlock * xpu::block_idx::x();

    experimental::CbmStsDigi* res = SortT(smem.sortBuf).sort(
        &data[offset], itemsPerBlock, &buf[offset],
        [](const experimental::CbmStsDigi& a) { return ((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time); }
    );

    if (xpu::block_idx::x() == 0) {
        *out = res;
    }
}

// ------------------------------------------------------------------------------------------------------------------------------

XPU_IMAGE(JanSergeySortKernel);

struct JanSergeySortSmem {
    int* countAndPrefixes;
};

XPU_KERNEL(JanSergeySort, JanSergeySortSmem, const int n, const experimental::CbmStsDigi* digis, const int* startIndex, const int* endIndex, experimental::CbmStsDigi* output) {
    const int bucketIdx = xpu::block_idx::x();
    const int bucketStartIdx = startIndex[bucketIdx];
    const int bucketEndIdx = endIndex[bucketIdx];
    const int threadStart = bucketStartIdx + xpu::thread_idx::x();

    __shared__ int countAndPrefixes[experimental::channelCount];

    // 1. Init all channel counters to zero: O(channelCount)
    for (int i = xpu::thread_idx::x(); i < experimental::channelCount; i += xpu::block_dim::x()) {
        countAndPrefixes[i] = 0;
    }

    __syncthreads();

    // 2. Count channels: O(n + channelCount)
    for (int i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        xpu::atomic_add_block(&countAndPrefixes[digis[i].channel], 1);
    }

    __syncthreads();

    // 3. Prefix sum computation, TODO: replace by cub: https://nvlabs.github.io/cub/structcub_1_1_device_scan.html
    prescan(countAndPrefixes);

    __syncthreads();

    // 4. Final sorting, place the elements in the correct position within the global output array: O(n)
    for (int i = bucketStartIdx + xpu::thread_idx::x(); i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        output[bucketStartIdx + xpu::atomic_add_block(&countAndPrefixes[digis[i].channel], 1)] = digis[i];
    }
}

__device__ void prescan(int* countAndPrefixes) {
    __shared__ int temp[experimental::channelCount];// allocated on invocation

    const int n = experimental::channelCount;
    const int countOffset = 0;

    int thid = threadIdx.x;
    int offset = 1;

    // load input into shared memory
    temp[2 * thid] = countAndPrefixes[countOffset + 2 * thid];
    temp[2 * thid + 1] = countAndPrefixes[countOffset + 2 * thid + 1];

    // build sum in place up the tree
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

     if (thid == 0) { temp[n - 1] = 0; } // clear the last element

     // traverse down tree & build scan
     for (int d = 1; d < n; d *= 2) {
         offset >>= 1;

         __syncthreads();

         if (thid < d) {
             int ai = offset * (2 * thid + 1) - 1;
             int bi = offset * (2 * thid + 2) - 1;
             float t = temp[ai];
             temp[ai] = temp[bi];
             temp[bi] += t;
         }
     }

     __syncthreads();

     countAndPrefixes[countOffset + 2 * thid] = temp[2 * thid]; // write results to device memory
     countAndPrefixes[countOffset + 2 * thid + 1] = temp[2 * thid + 1];
}

/*
  xpu::run_kernel<SortDigis>(xpu::grid::n_blocks(hfc.nModules * 2));
XPU_KERNEL(SortDigis, CbmStsSortDigiSmem) { xpu::cmem<HitFinder>().sortDigisInSpaceAndTime(smem, xpu::block_idx::x()); }

using SortDigisT = xpu::block_sort<unsigned long int, CbmStsDigi, xpu::block_size<SortDigis> {}, CBM_STS_SORT_ITEMS_PER_THREAD>;

struct CbmStsSortDigiSmem {
    // TODO no magic numbers for blocksize
    typename SortDigisT::storage_t sortBuf;
};

XPU_D void CbmStsGpuHitFinder::sortDigisInSpaceAndTime(CbmStsSortDigiSmem& smem, int iBlock) const
{
  int iModule          = iBlock;
  CbmStsDigi* digis    = &digisPerModule[digiOffsetPerModule[iModule]];
  CbmStsDigi* digisTmp = &digisPerModuleTmp[digiOffsetPerModule[iModule]];
  int nDigis           = getNDigis(iModule);

  SortDigisT digiSort(smem.sortBuf);

  digis = digiSort.sort(digis, nDigis, digisTmp, [](const CbmStsDigi a) {
    return ((unsigned long int) a.fChannel) << 32 | (unsigned long int) (a.fTime);
  });

  if (xpu::thread_idx::x() == 0) { digisSortedPerModule[iModule] = digis; }
}


XPU_KERNEL(SortDigis, CbmStsSortDigiSmem) { xpu::cmem<HitFinder>().sortDigisInSpaceAndTime(smem, xpu::block_idx::x()); }
*/
