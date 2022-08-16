#pragma once

#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"

#include <iostream>
#include <vector>

template<typename Kernel>
class blocksort_bench : public benchmark {

    const size_t n;
    size_t elems_per_block;
    const size_t n_blocks = 64; // seems hardcoded in block_sort

    // This is an internal copy of the unsorted digis
    // which is copied to run the sort benchmark repeatedly.
    experimental::CbmStsDigi* digis;
    experimental::CbmStsDigi* sorted;
    experimental::CbmStsDigiBucket* bucket;

    // Output pointer.
    experimental::CbmStsDigi** devOutput;
    experimental::CbmStsDigi* devBuffer; // Only used on device, not copied back to host.

    xpu::hd_buffer<experimental::CbmStsDigi> buffDigis;
    xpu::hd_buffer<int> buffStartIndex;
    xpu::hd_buffer<int> buffEndIndex;

    std::vector<float> timings_;

public:
    blocksort_bench(const experimental::CbmStsDigi* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true) : n(in_n), sorted(new experimental::CbmStsDigi[in_n]), digis(new experimental::CbmStsDigi[in_n]), benchmark(in_write, in_check) {
        // Create an internal copy of the digis.
        std::copy(in_digis, in_digis + in_n, digis);
        elems_per_block = n / n_blocks;
    }

    ~blocksort_bench() {}

    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        devOutput = xpu::device_malloc<experimental::CbmStsDigi*>(n);
        devBuffer = xpu::device_malloc<experimental::CbmStsDigi>(n);

        buffDigis = xpu::hd_buffer<experimental::CbmStsDigi>(n);

        bucket = new experimental::CbmStsDigiBucket(digis, n);
        std::cout << "BlockSort: Buckets created." << "\n";

        buffStartIndex = xpu::hd_buffer<int>(bucket->size());
        buffEndIndex = xpu::hd_buffer<int>(bucket->size());

        std::copy(bucket->startIndex, bucket->startIndex + bucket->size(), buffStartIndex.h());
        std::copy(bucket->endIndex, bucket->endIndex + bucket->size(), buffEndIndex.h());
        std::copy(bucket->digis, bucket->digis + n, buffDigis.h());
    }

    size_t size() override { return n; }

    void run() {
        xpu::copy(buffDigis, xpu::host_to_device);
        xpu::copy(buffStartIndex, xpu::host_to_device);
        xpu::copy(buffEndIndex, xpu::host_to_device);

        const auto started = std::chrono::high_resolution_clock::now();

        // const experimental::CbmStsDigi* data, const int* startIndex, const int* endIdex, experimental::CbmStsDigi* buf, experimental::CbmStsDigi** out, const size_t numElems
        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(bucket->size()), buffDigis.d(), buffStartIndex.d(), buffEndIndex.d(), devBuffer, devOutput, n);

        const auto done = std::chrono::high_resolution_clock::now();
        timings_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count());

        // Get the buffer that contains the sorted data.
        experimental::CbmStsDigi* hostOutput = nullptr;
        xpu::copy(&hostOutput, devOutput, 1);

        // Copy to back host.
        xpu::copy(sorted, hostOutput, n);
    }

    experimental::CbmStsDigi* output() override { return sorted; }

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

    void check() override {
        // Check if data is sorted.
        bool ok = true;

        // Start at second element and compare to previous for all i.
        for (size_t i = 1; i < n; i++) {
            bool okThisRun = true;

            const auto& curr = sorted[i];
            const auto& prev = sorted[i - 1];

            // Within the same address range, channel numbers increase.
            if (curr.address == prev.address) {
                ok &= curr.channel >= prev.channel;
                okThisRun &= curr.channel >= prev.channel;
            }
            if (curr.channel == prev.channel) {
                ok &= curr.time >= prev.time;
                okThisRun &= curr.time >= prev.time;
            }

            if (!okThisRun) {
                std::cout << "\nBlockSort Error: " << "\n";
                printf("(%lu/%lu): (%d, %d, %d)\n", i-1, n, prev.address, prev.channel, prev.time);
                printf("(%lu/%lu): (%d, %d, %d)\n", i, n, curr.address, curr.channel, curr.time);
            }
        }

        if (ok) {
            std::cout << "BlockSort: Data is sorted!" << std::endl;
        } else {
            std::cout << "BlockSort Error: Data is not sorted!" << std::endl;
        }
    }

    size_t bytes() { return n * sizeof(experimental::CbmStsDigi); }

    std::vector<float> timings() { return timings_; }

};
