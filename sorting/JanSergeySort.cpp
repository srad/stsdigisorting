#include <xpu/device.h>
#include "JanSergeySort.h"
#include "../datastructures.h"
#include "../common.h"

XPU_IMAGE(JanSergeySortKernel);

struct JanSergeySortSmem {
    unsigned int countAndPrefixes[experimental::channelCount];
    unsigned int temp[experimental::channelCount];
};

XPU_D void prescan(JanSergeySortSmem& smem);

XPU_KERNEL(JanSergeySort, JanSergeySortSmem, const size_t n, const experimental::CbmStsDigi* digis, const int* startIndex, const int* endIndex, experimental::CbmStsDigi* output) {
    const int bucketIdx = xpu::block_idx::x();
    const int bucketStartIdx = startIndex[bucketIdx];
    const int bucketEndIdx = endIndex[bucketIdx];
    const int threadStart = bucketStartIdx + xpu::thread_idx::x();

    // 1. Init all channel counters to zero: O(channelCount)
    // This step is not related to the input size.
    for (int i = xpu::thread_idx::x(); i < experimental::channelCount; i += xpu::block_dim::x()) {
        smem.countAndPrefixes[i] = 0;
    }

    xpu::barrier();

    // 2. Count channels: O(n + channelCount)
    for (int i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        xpu::atomic_add_block(&smem.countAndPrefixes[digis[i].channel], 1);
    }
    xpu::barrier();

    // 3. Prefix sum: value[i] = sum(0, i-1)
    prescan(smem);
    xpu::barrier();

    // 4. Final sorting, place the elements in the correct position within the global output array: O(n)
    if (xpu::thread_idx::x() == 0) {
        for (int i = bucketStartIdx; i <= bucketEndIdx && i < n; i++) {
            output[bucketStartIdx + smem.countAndPrefixes[digis[i].channel]] = digis[i];
            smem.countAndPrefixes[digis[i].channel]++;
        }
    }
}

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