#pragma once

#include "kernels.h"

namespace experimental {
    /// <summary>
    /// Launch for each bucket a block with one thread and run the sorting algorithm within each block linearly but among blocks in parallel on each bucket seperately.
    /// Each block will process a seperate chunk of the array (virtual buckets) and will not interfere with each other.
    /// The buckets are not evenly sized, but variable, depending on how many elements a bucket contains. startIndex[i] and endIndex[i] contains the bucket indexes.
    /// +----------+----------+-----+----------+    +---------+---------+-----+---------+
    /// | bucket 1 | bucket 2 | ... | bucket n | -> | block 0 | block 1 | ... | block n |
    /// +----------+----------+-----+----------+    +---------+---------+-----+---------+
    /// </summary>
    /// <param name="buckets"></param>
    /// <param name="bucketCount"></param>
    /// <param name="digiCount"></param>
    /// <param name="startIndex"></param>
    /// <param name="endIndex"></param>
    /// <returns></returns>
    cudaError_t janSergeySort(CbmStsDigi* buckets, const int bucketCount, const int digiCount, const int* startIndex, const int* endIndex) {
        CbmStsDigi* dev_buckets = 0;
        int* dev_startIndex = 0;
        int* dev_endIndex = 0;

        CbmStsDigi* dev_output = 0;
        int* dev_countAndPrefixes = 0;

        try {
            const int digiSize = digiCount * sizeof(CbmStsDigi);
            const int bucketSize = bucketCount * sizeof(int); // startIndex, endIndex

            // Each bucket has a seperate digi channel counter.
            // This array is not really big, channelCount = 2048 and the number of buckets was limited in test runs (~1000).
            const int countAndPrefixesSize = bucketCount * channelCount * sizeof(int);

            CHECK_ERROR(cudaSetDevice(0));

            // 1. Reserve memory on device and copy data to device.
            CHECK_ERROR(cudaMalloc((void**) &dev_buckets, digiSize));
            CHECK_ERROR(cudaMemcpy(dev_buckets, buckets, digiSize, cudaMemcpyHostToDevice));

            // Temp memory for sorted output. The input is copied to this output.
            // After temp copy the data is copied to the original bucket, by overwriting the elements in each bucket.
            CHECK_ERROR(cudaMalloc((void**) &dev_output, digiSize));

            CHECK_ERROR(cudaMalloc((void**) &dev_startIndex, bucketSize));
            CHECK_ERROR(cudaMemcpy(dev_startIndex, startIndex, bucketSize, cudaMemcpyHostToDevice));

            CHECK_ERROR(cudaMalloc((void**) &dev_endIndex, bucketSize));
            CHECK_ERROR(cudaMemcpy(dev_endIndex, endIndex, bucketSize, cudaMemcpyHostToDevice));

            CHECK_ERROR(cudaMalloc((void**) &dev_countAndPrefixes, countAndPrefixesSize));

            constexpr int threadsPerBucket = 1024;

            // 2. Launch a kernel on the GPU with one thread for each bucket.
            cudaMemset(dev_countAndPrefixes, 0, countAndPrefixesSize);
            CHECK_ERROR(cudaDeviceSynchronize());
            CHECK_ERROR(cudaGetLastError());

            countChannels << < bucketCount, threadsPerBucket >> > (digiCount, dev_buckets, dev_countAndPrefixes, dev_startIndex, dev_endIndex);
            //countChannels << <bucketCount, threadsPerBucket >> > (digiCount, dev_buckets, dev_countAndPrefixes, dev_startIndex, dev_endIndex);
            CHECK_ERROR(cudaDeviceSynchronize());
            CHECK_ERROR(cudaGetLastError());

            //prescan << <bucketCount, 1024 >> > (dev_countAndPrefixes);
            computePrefixSum << < bucketCount, 1 >> > (digiCount, dev_countAndPrefixes);
            CHECK_ERROR(cudaDeviceSynchronize());
            CHECK_ERROR(cudaGetLastError());

            sortKernelParallel << < bucketCount, 1024 >> > (digiCount, dev_buckets, dev_output, dev_countAndPrefixes, bucketCount, dev_startIndex, dev_endIndex);
            CHECK_ERROR(cudaDeviceSynchronize());
            CHECK_ERROR(cudaGetLastError());

            // 3. Copy result to host.
            CHECK_ERROR(cudaMemcpy(buckets, dev_output, digiSize, cudaMemcpyDeviceToHost));
        }
        catch (thrust::system_error& e) {
            std::cerr << "CUDA error during some_function: " << e.what() << std::endl;
        }
        catch (std::bad_alloc& e) {
            std::cerr << "Bad memory allocation during some_function: " << e.what() << std::endl;
        }
        catch (std::runtime_error& e) {
            std::cerr << "Runtime error during some_function: " << e.what() << std::endl;
        }
        catch (std::exception e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        cudaFree(dev_countAndPrefixes);
        cudaFree(dev_output);
        cudaFree(dev_buckets);
        cudaFree(dev_startIndex);
        cudaFree(dev_endIndex);

        return cudaGetLastError();
    }
}
