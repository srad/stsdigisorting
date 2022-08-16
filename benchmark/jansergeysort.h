#pragma once

#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"
#include <iostream>
#include <vector>
#include <chrono>

template<typename Kernel>
class jansergeysort_bench : public benchmark {

    const size_t n;

    std::vector<float> timings_;

    // Big difference here is that the digis are grouped in buckets and then bucket-wise sorted.
    experimental::CbmStsDigiBucket* bucket;

    experimental::CbmStsDigi* digis;
    xpu::hd_buffer <experimental::CbmStsDigi> buffDigis;
    xpu::hd_buffer <experimental::CbmStsDigi> buffOutput;

    xpu::hd_buffer<int> buffStartIndex;
    xpu::hd_buffer<int> buffEndIndex;

public:
    jansergeysort_bench(const experimental::CbmStsDigi* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true) : n(in_n), digis(new experimental::CbmStsDigi[in_n]), benchmark(in_write, in_check) {
        std::copy(in_digis, in_digis + in_n, digis);
    }

    ~jansergeysort_bench() {}

    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        buffDigis = xpu::hd_buffer<experimental::CbmStsDigi>(n);
        std::copy(digis, digis + n, buffDigis.h());
        
        buffOutput = xpu::hd_buffer<experimental::CbmStsDigi>(n);

        bucket = new experimental::CbmStsDigiBucket(digis, n);
        std::cout << "Buckets created." << "\n";

        buffStartIndex = xpu::hd_buffer<int>(bucket->size());
        buffEndIndex = xpu::hd_buffer<int>(bucket->size());

        std::copy(bucket->startIndex, bucket->startIndex + bucket->size(), buffStartIndex.h());
        std::copy(bucket->endIndex, bucket->endIndex + bucket->size(), buffEndIndex.h());
        std::copy(bucket->digis, bucket->digis + n, buffDigis.h());
    }

    void teardown() {
        delete[] digis;
        delete bucket;
        buffDigis.reset();
        buffOutput.reset();
    }

    size_t size_n() const { return n; }

    void run() {
        xpu::copy(buffDigis, xpu::host_to_device);

        xpu::copy(buffStartIndex, xpu::host_to_device);
        xpu::copy(buffEndIndex, xpu::host_to_device);

        const auto started = std::chrono::high_resolution_clock::now();

        // For now, one block per bucket.
        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(bucket->size()), n, buffDigis.d(), buffStartIndex.d(), buffEndIndex.d(), buffOutput.d());

        const auto done = std::chrono::high_resolution_clock::now();
        timings_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count());

        // Copy result back to host.
        xpu::copy(buffOutput, xpu::device_to_host);
    }

    size_t size() override { return n; }

    experimental::CbmStsDigi* output() override { return buffOutput.h(); }

    void write() override {
        benchmark::write();
        
        // +------------------------------------------------------------------------------+
        // |                               Bucket data                                    |
        // +------------------------------------------------------------------------------+
        std::ofstream bucketOutput;
        bucketOutput.open("jan_sergey_bucket.csv", std::ios::out | std::ios::trunc);
        bucketOutput << "bucket,index,address,channel,time\n";

        for(int i=0; i < bucket->size(); i++) {
            const auto start = bucket->startIndex[i];
            const auto end   = bucket->endIndex[i];
            for(int j=start; j <= end; j++) {
                bucketOutput << i << "," << j << "," << bucket->digis[j].address << "," << bucket->digis[j].channel << "," << bucket->digis[j].time  << "\n";
            }
        }
        bucketOutput.close();
    }

    size_t bytes() { return n * sizeof(experimental::CbmStsDigi); }

    std::vector<float> timings() { return timings_; /*return xpu::get_timing<Kernel>();*/ }

};
