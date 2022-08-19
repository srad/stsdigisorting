#include <xpu/device.h>
#include "JanSergeySortSided.h"
#include "../datastructures.h"
#include "../common.h"
#include "device_fun.h"

XPU_IMAGE(JanSergeySortSidedKernel);

struct JanSergeySortSidedSmem {
    unsigned int prefixSum[experimental::channelCount / 2];
    // Digi channels in range [1024, 2047] must be placed after all digis with the range [0, 1023].
    // This variable counts the offset for all digis in range [1024, 2047] for the odd blocks.
    unsigned int digiCountOnFront;
    unsigned int temp[experimental::channelCount / 2];
};

XPU_KERNEL(JanSergeySortSided, JanSergeySortSidedSmem, const size_t n, const experimental::CbmStsDigi* digis, const unsigned int* startIndex, const unsigned int* endIndex, experimental::CbmStsDigi* output, unsigned int* sideStartIndex, unsigned int* sideEndIndex) {
    // Two blocks handle one bucket.
    // 0, 0, 1, 1, 2, 2, ...
    const int bucketIdx = xpu::block_idx::x() / 2;
    const int bucketStartIdx = startIndex[bucketIdx];
    const int bucketEndIdx = endIndex[bucketIdx];

    // Each thread starts at a different offset and increments by xpu::block_dim::x(),
    // so two threads do not handle the same item.
    const int threadStart = bucketStartIdx + xpu::thread_idx::x();

    // 0, 1, 0, 1, ...
    const unsigned int blockSide = xpu::block_idx::x() % 2; // 0=front, 1=back   
    const bool isFront = blockSide == 0;
    const bool isBack = blockSide == 1;

    // +--------------------------------------------------------------------+
    // | Bucket 0             | Bucket 1          | Bucket 2          |
    // +---------+------------+---------+------------+---------+------------+
    // | Block 0 | Block 1    | Block 2 | Block 3    | Block 4 | Block 5    |
    // +---------+------------+---------+------------+---------+------------+
    // | 0..1023 | 1024..2047 | 0..1023 | 1024..2047 | 0..1023 | 1024..2047 |
    // +---------+------------+---------+------------+---------+------------+
    // | Front   | Back       |  Front  | Back       | Front   | Back       |
    // +---------+------------+---------+------------+---------+------------+
    //
    // Even blocks sort the front, odd blocks the back.

    // 1. Init all channel counters to zero: O(channelCount)
    // This step is not related to the input size.
    if (xpu::thread_idx::x() == 0) {
        smem.digiCountOnFront = 0;
    }

    // 1024 threads per block and each block handles 1024 channels, each index will be initialized.
    smem.prefixSum[xpu::thread_idx::x()] = 0;
    xpu::barrier();

    // 2. Count channels: O(n + channelCount)
    // --------------------------------------
    // Depending on the channel count it eather in sideCounter[front] or sideCounter[back]:
    // address	  channel	time
    // -------------------------
    // 268735554	 983	1792 => Block 0: (983-1024 > 0) => side 0 
    // 268502018	1630	1793 => Block 0: (1630-1024 > 1) => side 1
    // 268664834     107	1793 ...
    for (int i = threadStart; i <= bucketEndIdx && i < n; i += xpu::block_dim::x()) {
        const bool countFront = isFront && (digis[i].channel < 1024);
        const bool countBack = isBack && (digis[i].channel >= 1024);
        const unsigned int count = countFront || countBack;

        xpu::atomic_add_block(&smem.prefixSum[digis[i].channel % 1024], count);
        xpu::atomic_add_block(&smem.digiCountOnFront, (digis[i].channel < 1024));
    }
    xpu::barrier();

    // 3. Prefix sum computation, TODO: replace by cub: https://nvlabs.github.io/cub/structcub_1_1_device_scan.html
    //prescan(smem.prefixSum, smem.temp); (doesnt work with this arragement)
    if (xpu::thread_idx::x() == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < experimental::channelCount / 2; i++) {
            const auto tmp = smem.prefixSum[i];
            smem.prefixSum[i] = sum;
            sum += tmp;
        }
    }
    xpu::barrier();
    
    //if (xpu::thread_idx::x() < 2 && xpu::block_idx::x() < 2) {
    //    printf("threadidx=%d blockidx=%d start=%d end=%d size=%d digiCountPerChannelFront=%d\n", xpu::thread_idx::x(), xpu::block_idx::x(), bucketStartIdx, bucketEndIdx, bucketEndIdx-bucketStartIdx+1, smem.digiCountPerChannelFront);
    //}
    // block=0 => 0
    // block=1 => sideCount[0]

    // 4. Final sorting, place the elements in the correct position within the global output array: O(n)
    // Front: blockSide=0 stop at the length of its digis
    // Back:  blockSide=1 stop at bucketEndIdx
    const unsigned int sideStartIdx = bucketStartIdx + ((smem.digiCountOnFront) * isBack);
    const unsigned int sideEndIdx = (isBack * bucketEndIdx) + ((sideStartIdx + smem.digiCountOnFront - 1) * isFront);

    sideStartIndex[2 * bucketIdx + blockSide] = sideStartIdx;
    sideEndIndex[2 * bucketIdx + blockSide] = sideEndIdx;
    
    //if (xpu::block_idx::x() < 4 && xpu::thread_idx::x() < 2) {
     //   printf("threadidx=%d blockidx=%d start=%d end=%d size=%d sideStartIdx=%d sideEndIdx=%d digiCountOnFront=%d blockSide=%d channelStartIdx=%d channelEndIdx=%d\n", xpu::thread_idx::x(), xpu::block_idx::x(), bucketStartIdx, bucketEndIdx, bucketEndIdx-bucketStartIdx+1, sideStartIdx, sideEndIdx, smem.digiCountOnFront, blockSide, channelStartIdx, channelEndIdx);
    //}

    // All threads traverse all digis and only sort the digis which are relevant for them.
    //for (int i = threadStart, j=0; i <= bucketEndIdx && i < n; i += xpu::block_dim::x(), j++) {
    for (int i = bucketStartIdx; i <= bucketEndIdx && i < n; i++) {
        //if (xpu::thread_idx::x() < 2 && xpu::block_idx::x() < 5) {
        //    printf("threadidx=%d blockidx=%d blockdim=%d i=%d j=%d start=%d end=%d size=%d\n", xpu::thread_idx::x(), xpu::block_idx::x(), xpu::block_dim::x(), i, j, bucketStartIdx, bucketEndIdx, bucketEndIdx-bucketStartIdx+1);
        //}
        // Even blocks sort all digis from channel: 0..1023
        // Odd blocks sort all digis from channel:  1024..2047
        // So in each block one thread is responsible for all digis where channel == threadId.x
        
        // Thread ids go from 0..1023
        const bool handledByThisThread = (digis[i].channel % 1024) == xpu::thread_idx::x();
        if (handledByThisThread) {
            const bool handledByThisBlock = (isFront && (digis[i].channel < 1024)) || (isBack && (digis[i].channel >= 1024));
            if (handledByThisBlock) {
                output[sideStartIdx + (smem.prefixSum[digis[i].channel % 1024]++)] = digis[i];
            }
        }
    }
}