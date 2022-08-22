#include <xpu/device.h>
#include "JanSergeySort.h"
#include "../datastructures.h"
#include "../common.h"
#include "device_fun.h"

XPU_IMAGE(JanSergeySortKernel);

struct JanSergeySortSmem {
    unsigned int countAndPrefixes[experimental::channelCount];
    //unsigned int temp[experimental::channelCount];
    //unsigned int output[experimental::channelCount];
};

constexpr int TPB = 128;
constexpr int IPT = experimental::channelCount / TPB;

XPU_KERNEL(JanSergeySort, JanSergeySortSmem, const size_t n, const experimental::CbmStsDigi* digis, const int* startIndex, const int* endIndex, experimental::CbmStsDigi* output, unsigned int* sideStartIndex, unsigned int* sideEndIndex) {
    const int bucketIdx = xpu::block_idx::x();
    const int bucketStartIdx = startIndex[bucketIdx];
    const int bucketEndIdx = endIndex[bucketIdx];
    const int threadStart = bucketStartIdx + xpu::thread_idx::x();

    // -----------------------------------------------------------------------------------------------------------
    // 1. Init all channel counters to zero: O(channelCount)
    // This step is not related to the input size.
    // -----------------------------------------------------------------------------------------------------------
    for (int i = xpu::thread_idx::x(); i < experimental::channelCount; i += xpu::block_dim::x()) {
        smem.countAndPrefixes[i] = 0;
    }
    xpu::barrier();

    // -----------------------------------------------------------------------------------------------------------
    // 2. Count channels: O(n + channelCount) -> O(n)
    // -----------------------------------------------------------------------------------------------------------
    for (int i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        xpu::atomic_add_block(&smem.countAndPrefixes[digis[i].channel], 1);
    }
    xpu::barrier();

    // -----------------------------------------------------------------------------------------------------------
    // |3. Prefix sum: value[i] = sum(0, i-1) -> O(channelCount)
    // -----------------------------------------------------------------------------------------------------------
    //prescan(smem.countAndPrefixes, smem.temp);

    // Alternative:
    /*
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
    */
    //xpu::barrier();

    // -----------------------------------------------------------------------------------------------------------
    // 4. Final sorting, place the elements in the correct position within the global output array: O(n)
    //
    // This must be done linearly otherwise there will be race conditions if a thread on the right wants to insert
    // a digis into the same channel as a thread on the left. Might be handled by some algoritm, unclear yet.
    // -----------------------------------------------------------------------------------------------------------
    if (xpu::thread_idx::x() == 0) {
        // TODO: replace by cub::DeviceScan::ExclusiveSum (https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#a02b2d2e98f89f80813460f6a6ea1692b)
        unsigned int sum = 0;
        for (int i = 0; i < experimental::channelCount; i++) {
            const auto tmp = smem.countAndPrefixes[i];
            smem.countAndPrefixes[i] = sum;
            sum += tmp;
        }
        
        // Front
        sideStartIndex[bucketIdx * 2] = bucketStartIdx;
        sideEndIndex[  bucketIdx * 2] =  bucketStartIdx + smem.countAndPrefixes[1024] - 1;

        // Back
        sideStartIndex[bucketIdx * 2 + 1] = sideEndIndex[  bucketIdx * 2] + 1;
        sideEndIndex[  bucketIdx * 2 + 1] = bucketEndIdx;

        for (int i = bucketStartIdx; i <= bucketEndIdx && i < n; i++) {
            output[bucketStartIdx + (smem.countAndPrefixes[digis[i].channel]++)] = digis[i];
        }
    }
}