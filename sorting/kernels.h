#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "../common.h"

namespace experimental {
#define DEBUG

#ifdef DEBUG
# define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

    // CUDA INFO:
    // ---------
    // blockDim.x, y, z                      gives the number of threads in a block, in the particular direction
    // gridDim.x, y, z                       gives the number of blocks in a grid, in the particular direction
    // blockDim.x* gridDim.x                 gives the number of threads in a grid(in the x direction, in this
    // threadIdx.x + blockIdx.x * blockDim.x gives global thread id
    // threadIdx.x: thread offset within each block, so:
    //              +---+---+---+---+---+---+---+---+
    // idx:         | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
    //              +---+---+---+---+---+---+---+---+
    // threadIdx.x: | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |
    //              +---+---+---+---+---+---+---+---+
    // blockIdx.x:  |       0       |       1       |
    //              +---------------+---------------+

    __global__ void countChannelsSingleThread(const int n, CbmStsDigi *digis, int *countAndPrefixes, int *startIndex, int *endIndex) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < n) {
            const int digisStartIdx = startIndex[idx]; // Bucket start index.
            const int digisEndIdx = endIndex[idx]; // including this index is the end of the bucket.

            const int countOffsetStartIdx = idx * channelCount;
            const int countOffsetEndIdx = countOffsetStartIdx + channelCount;

            // 1. Init count array to zero. If multiple threads per block, turn into: __shared__ int countAndPrefixSum[channelCount];
            for (int i = countOffsetStartIdx; i < countOffsetEndIdx; i++) {
                countAndPrefixes[i] = 0;
            }

            // 2. Count the channels in the digi bucket from begin to end.
            for (int i = digisStartIdx; i <= digisEndIdx; i++) {
                const int channelOffset = countOffsetStartIdx + digis[i].channel;
                countAndPrefixes[channelOffset] += 1;
            }
        }
    }

    /// <summary>
    /// 1. Init + 2. Count channels, per bucket.
    /// The counting array is one continous array of size bucketCount * channelCount.
    /// Each block is assigned to a bucket, hence the "number of buckets" == "number of block", on the GPU.
    /// Within each block (or bucket) a number of threads can now count in parallel the channels.
    ///
    /// Performance waste: Since the buckets have variable length,
    ///                    the number of threads per block could be much bigger than the number of elements in the bucket.
    /// </summary>
    /// <param name="n"></param>
    /// <param name="digis"></param>
    /// <param name="countAndPrefixes"></param>
    /// <param name="startIndex"></param>
    /// <param name="endIndex"></param>
    /// <returns></returns>
    __global__ void countChannels(const int n, CbmStsDigi *digis, int *countAndPrefixes, const int *startIndex, const int *endIndex) {
        const int bucketIdx = blockIdx.x;
        const int bucketStartIdx = startIndex[bucketIdx];
        const int bucketEndIdx = endIndex[bucketIdx];
        const int threadStart = bucketStartIdx + threadIdx.x;

        for (int i = threadStart; i <= bucketEndIdx && i < n; i += blockDim.x) {
            const int channelOffset = bucketIdx * channelCount + digis[i].channel;
            atomicAdd(&countAndPrefixes[channelOffset], 1);
        }
    }

    //cub prefix sum algo nehmen

    /// <summary>
    /// Computes index offsets through the count of the previous elements.
    /// Counts: [2, 4, 3] => Offsets: [0, 2, 6]
    /// </summary>
    /// <param name="countAndPrefixes"></param>
    /// <param name="countOffsetStartIdx"></param>
    /// <param name="countOffsetEndIdx"></param>
    /// <returns></returns>
    __global__ void computePrefixSum(const int n, int *countAndPrefixes) {
        // Compute (partial) prefix sum: Overwrite the existing values.
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < n) {
            const int countOffsetStartIdx = idx * channelCount;
            const int countOffsetEndIdx = countOffsetStartIdx + channelCount;

            int sum = 0;
            for (int i = countOffsetStartIdx; i < countOffsetEndIdx; i++) {
                const auto tmp = countAndPrefixes[i];
                countAndPrefixes[i] = sum;
                sum += tmp;
            }
        }
    }

    /// <summary>
    /// For details about this algorighm see: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    /// </summary>
    /// <param name="countAndPrefixes"></param>
    /// <returns></returns>
    __global__ void prescan(int *countAndPrefixes) {
        __shared__ float temp[channelCount];// allocated on invocation

        const int n = channelCount;
        const int bucketIdx = blockIdx.x;
        const int countOffset = blockIdx.x * channelCount;

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

    /// <summary>
    /// This counting kernel uses shared memory, but is roughly a little slower than the
    /// non shared memory version, since it required two synchronisations and an array copy, instead of one atomicAdd.
    /// </summary>
    /// <param name="n"></param>
    /// <param name="digis"></param>
    /// <param name="countAndPrefixes"></param>
    /// <param name="startIndex"></param>
    /// <param name="endIndex"></param>
    /// <returns></returns>
    __global__ void countChannelsSharedMemory(const int n, CbmStsDigi *digis, int *countAndPrefixes, const int *startIndex, const int *endIndex) {
        __shared__ unsigned int cnt[channelCount];

        const int bucketIdx = blockIdx.x;
        const int bucketStartIdx = startIndex[bucketIdx];
        const int bucketEndIdx = endIndex[bucketIdx];
        const int threadStart = bucketStartIdx + threadIdx.x;

        // In the regular version this is not needed, since cudaMemset() is used to initilize the values.
        for (int i = threadIdx.x; i < channelCount; i += blockDim.x) {
            cnt[i] = 0;
        }
        __syncthreads();

        // Count within the shared memory.
        for (int i = threadStart; i <= bucketEndIdx && i < n; i += blockDim.x) {
            atomicAdd(&cnt[digis[i].channel], 1);
        }
        __syncthreads();

        // Just set the parallel counted channels at the right array segment.
        for (int i = threadIdx.x; i < channelCount; i += blockDim.x) {
            countAndPrefixes[(bucketIdx * channelCount) + i] = cnt[i];
        }
    }

    __global__ void sortKernelParallel(const int n, CbmStsDigi *digis, CbmStsDigi *output, int *countAndPrefixes, const int bucketCount, const int *startIndex, const int *endIndex) {
        const int bucketIdx = blockIdx.x;
        const int bucketStartIdx = startIndex[bucketIdx];
        const int bucketEndIdx = endIndex[bucketIdx];
        const int bucketOffset = bucketStartIdx + threadIdx.x;

        /*
        __shared__ unsigned int channelOffsetCounter[channelCount];

        for (int i = threadIdx.x; i < channelCount; i += blockDim.x) {
            channelOffsetCounter[i] = 0;
        }

        __syncthreads();

        const int start = bucketStartIdx + threadIdx.x * blockDim.x;
        const int end = start + blockDim.x;

        for (int i = start; i <= bucketEndIdx && i < end && i < n; i++) {
            const int channelOffset = bucketIdx * channelCount + digis[i].channel;
            output[bucketStartIdx + countAndPrefixes[channelOffset] + channelOffsetCounter[digis[i].channel]] = digis[i];
            atomicAdd(&channelOffsetCounter[digis[i].channel], 1);
        }

        __syncthreads();
        */

        for (int i = bucketOffset; i <= bucketEndIdx && i < n; i += blockDim.x) {
            const int channelOffset = bucketIdx * channelCount + digis[i].channel;
            output[bucketStartIdx + atomicAdd(&countAndPrefixes[channelOffset], 1)] = digis[i];
        }
    }

    /// <summary>
    /// Notice that each bucket need to be placed linearly by one thread in the sorted output array.
    /// Otherwise a race condition in the offset counter.
    /// However, it might be possible to used the threadIdx.x to place the elements for each bucket in parallel.
    /// </summary>
    /// <param name="digis"></param>
    /// <param name="bucketCount"></param>
    /// <param name="startIndex"></param>
    /// <param name="endIndex"></param>
    /// <returns></returns>
    __global__ void sortKernel(const int n, CbmStsDigi *digis, CbmStsDigi *tmp, int *countAndPrefixes, const int bucketCount, const int *startIndex, const int *endIndex) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < n) {
            const int bucketStartIdx = startIndex[idx]; // Bucket start index.
            const int bucketEndIdx = endIndex[idx]; // including this index is the end of the bucket.

            // Sort elements by placing them at the right location in the temporary array.
            for (int i = bucketStartIdx; i <= bucketEndIdx; i++) {
                const int channelOffset = idx * channelCount + digis[i].channel;
                tmp[bucketStartIdx + countAndPrefixes[channelOffset]] = digis[i];
                countAndPrefixes[channelOffset] += 1;
            }

            // Copy/overwrite the buckets (section) with the sorted order.
            for (int i = bucketStartIdx; i <= bucketEndIdx; i++) {
                digis[i] = tmp[i];
            }
        }
    }
}
