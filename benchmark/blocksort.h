#pragma once

#include "../types.h"
#include "../datastructures.h"
#include "../constants.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"

#include <iostream>
#include <vector>

namespace experimental {

    template<typename Kernel>
    class blocksort_bench : public benchmark {

        const size_t n;
        size_t elems_per_block;
        const size_t n_blocks = 64; // seems hardcoded in block_sort

        // This is an internal copy of the unsorted digis
        // which is copied to run the sort benchmark repeatedly.
        CbmStsDigiInput* digis;
        digi_t* sorted;
        bucket_t* bucket;

        digi_t** devOutput;
        digi_t* devBuffer; // Only used on device, not copied back to host.

        xpu::hd_buffer<digi_t> buffDigis;
        xpu::hd_buffer<index_t> buffStartIndex;
        xpu::hd_buffer<index_t> buffEndIndex;

    public:
        blocksort_bench(const CbmStsDigiInput* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true) : n(in_n), sorted(new digi_t[in_n]), digis(new CbmStsDigiInput[in_n]), benchmark(in_write, in_check) {
            // Create an internal copy of the digis.
            std::copy(in_digis, in_digis + in_n, digis);
            elems_per_block = n / n_blocks;
        }

        ~blocksort_bench() {}

        std::string name() {
            return "BlockSort(" + std::to_string(BlockSortBlockDimX) + "," + std::to_string(BlockSortItemsPerThread) + "," + get_device() + ")";
        }

        void setup() {
            devOutput = xpu::device_malloc<digi_t*>(n);
            devBuffer = xpu::device_malloc<digi_t>(n);

            buffDigis = xpu::hd_buffer<digi_t>(n);

            bucket = new CbmStsDigiBucket(digis, n);
            std::cout << "BlockSort: Buckets created." << "\n";

            buffStartIndex = xpu::hd_buffer<index_t>(bucket->size());
            buffEndIndex = xpu::hd_buffer<index_t>(bucket->size());

            std::copy(bucket->startIndex, bucket->startIndex + bucket->size(), buffStartIndex.h());
            std::copy(bucket->endIndex, bucket->endIndex + bucket->size(), buffEndIndex.h());
            std::copy(bucket->digis, bucket->digis + n, buffDigis.h());
        }

        size_t size() override { return n; }

        void run() {
            xpu::copy(buffDigis, xpu::host_to_device);
            xpu::copy(buffStartIndex, xpu::host_to_device);
            xpu::copy(buffEndIndex, xpu::host_to_device);

            xpu::run_kernel<Kernel>(xpu::grid::n_blocks(bucket->size()), buffDigis.d(), buffStartIndex.d(), buffEndIndex.d(), devBuffer, devOutput, n);

            // Get the buffer that contains the sorted data.
            digi_t* hostOutput = nullptr;
            xpu::copy(&hostOutput, devOutput, 1);

            // Copy to back host.
            xpu::copy(sorted, hostOutput, n);
        }

        std::vector<float> timings() override { return xpu::get_timing<Kernel>(); }

        digi_t* output() override { return sorted; }

        void teardown() {
            delete[] digis;
            delete bucket;
            delete [] sorted;
            buffDigis.reset();
            buffStartIndex.reset();
            buffEndIndex.reset();
            xpu::free(devOutput);
            xpu::free(devBuffer);
        }

        size_t bytes() { return n * sizeof(digi_t); }

    };
    
}