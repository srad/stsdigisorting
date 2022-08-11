#include <xpu/device.h>
#include "JanSergeySort.h"
#include "../datastructures.h"
#include "../common.h"

XPU_IMAGE(JanSergeySortKernel);

struct JanSergeySortSmem {
    unsigned int countAndPrefixes[experimental::channelCount];
    //unsigned int temp[experimental::channelCount];
};

XPU_D void prescan(JanSergeySortSmem& smem);

XPU_KERNEL(JanSergeySort, JanSergeySortSmem, const size_t n, const experimental::CbmStsDigi* digis, const int* startIndex, const int* endIndex, experimental::CbmStsDigi* output) {
    const int bucketIdx = xpu::block_idx::x();
    const int bucketStartIdx = startIndex[bucketIdx];
    const int bucketEndIdx = endIndex[bucketIdx];
    const int threadStart = bucketStartIdx + xpu::thread_idx::x();

    // 1. Init all channel counters to zero: O(channelCount)
    for (int i = xpu::thread_idx::x(); i < experimental::channelCount; i += xpu::block_dim::x()) {
        smem.countAndPrefixes[i] = 0;
    }

    xpu::barrier();

    // 2. Count channels: O(n + channelCount)
    for (int i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        xpu::atomic_add_block(&smem.countAndPrefixes[digis[i].channel], 1);
    }

    xpu::barrier();

    // 3. Prefix sum computation, TODO: replace by cub: https://nvlabs.github.io/cub/structcub_1_1_device_scan.html
    //prescan(smem);

    // This computation is so small and not dependent on the input, that it might not even be worth optimising.
    // It is of length 2048 for each block and traverses the array linearly without addition space.
    if (xpu::thread_idx::x() == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < experimental::channelCount; i++) {
            const auto tmp = smem.countAndPrefixes[i];
            smem.countAndPrefixes[i] = sum;
            sum += tmp;
        }
    }

    xpu::barrier();

    // 4. Final sorting, place the elements in the correct position within the global output array: O(n)
    for (int i = bucketStartIdx + xpu::thread_idx::x(); i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        output[bucketStartIdx + xpu::atomic_add_block(&smem.countAndPrefixes[digis[i].channel], 1)] = digis[i];
    }
}

/*
XPU_D void prescan(JanSergeySortSmem& smem) {
    const int n = experimental::channelCount;
    const int countOffset = 0;

    int thid = xpu::thread_idx::x();
    int offset = 1;

    // load input into shared memory
    smem.temp[2 * thid] = smem.countAndPrefixes[countOffset + 2 * thid];
    smem.temp[2 * thid + 1] = smem.countAndPrefixes[countOffset + 2 * thid + 1];

    // build sum in place up the tree
    for (int d = n >> 1; d > 0; d >>= 1) {
        xpu::barrier();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            smem.temp[bi] += smem.temp[ai];
        }
        offset *= 2;
    }

     if (thid == 0) { smem.temp[n - 1] = 0; } // clear the last element

     // traverse down tree & build scan
     for (int d = 1; d < n; d *= 2) {
         offset >>= 1;

         xpu::barrier();

         if (thid < d) {
             int ai = offset * (2 * thid + 1) - 1;
             int bi = offset * (2 * thid + 2) - 1;
             float t = smem.temp[ai];
             smem.temp[ai] = smem.temp[bi];
             smem.temp[bi] += t;
         }
     }

     xpu::barrier();

     smem.countAndPrefixes[countOffset + 2 * thid] = smem.temp[2 * thid]; // write results to device memory
     smem.countAndPrefixes[countOffset + 2 * thid + 1] = smem.temp[2 * thid + 1];
}
*/

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
