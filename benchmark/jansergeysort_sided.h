#pragma once

#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"
#include <iostream>
#include <vector>
#include <chrono>

template<typename Kernel>
class jansergeysort_sided_bench : public benchmark {

    const size_t n;
    const unsigned int block_per_bucket;

    // Big difference here is that the digis are grouped in buckets and then bucket-wise sorted.
    experimental::CbmStsDigiBucket* bucket;

    experimental::CbmStsDigiInput* digis;
    xpu::hd_buffer <experimental::CbmStsDigi> buffDigis;
    xpu::hd_buffer <experimental::CbmStsDigi> buffOutput;

    xpu::hd_buffer<unsigned int> buffStartIndex;
    xpu::hd_buffer<unsigned int> buffEndIndex;

    xpu::hd_buffer<unsigned int> sideStartIndex;
    xpu::hd_buffer<unsigned int> sideEndIndex;

public:
    jansergeysort_sided_bench(const experimental::CbmStsDigiInput* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true, unsigned int in_block_per_bucket = 1) : n(in_n), digis(new experimental::CbmStsDigiInput[in_n]), block_per_bucket(in_block_per_bucket), benchmark(in_write, in_check) {
        std::copy(in_digis, in_digis + in_n, digis);
        std::cout << "(" << name() << ")" << " Block per bucket=" << block_per_bucket << "\n";
    }

    ~jansergeysort_sided_bench() {}

    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        buffDigis = xpu::hd_buffer<experimental::CbmStsDigi>(n);       

        // Sorted result is written to here.
        buffOutput = xpu::hd_buffer<experimental::CbmStsDigi>(n);

        bucket = new experimental::CbmStsDigiBucket(digis, n);
        std::cout << "Buckets created." << "\n";

        buffStartIndex = xpu::hd_buffer<unsigned int>(bucket->size());
        buffEndIndex = xpu::hd_buffer<unsigned int>(bucket->size());

        // sideStartIndex[2 * i    ] = front start index
        // sideStartIndex[2 * i + 1] = back start index
        // same for end index..
        sideStartIndex = xpu::hd_buffer<unsigned int>(bucket->size() * 2);
        sideEndIndex = xpu::hd_buffer<unsigned int>(bucket->size() * 2);

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

        auto t0 = std::chrono::high_resolution_clock::now();  

        // For now, one block per bucket.
        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(bucket->size() * block_per_bucket), n, buffDigis.d(), buffStartIndex.d(), buffEndIndex.d(), buffOutput.d(), sideStartIndex.d(), sideEndIndex.d());

        const auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t1 - t0;
        const auto durationMS = fp_ms.count();
        timings_.push_back(durationMS);

        // Copy result back to host.
        xpu::copy(buffOutput, xpu::device_to_host);

        xpu::copy(sideStartIndex, xpu::device_to_host);
        xpu::copy(sideEndIndex, xpu::device_to_host);

        for(int i=0; i < bucket->size(); i++) {
            printf("Bucket idx=%d front=(%d, %d) back=(%d, %d) \n", i, sideStartIndex.h()[2*i], sideEndIndex.h()[2*i], sideStartIndex.h()[2*i + 1], sideEndIndex.h()[2*i + 1]);
        }
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
                bucketOutput << i << "," << j << "," << bucket->digis[j].channel << "," << bucket->digis[j].time  << "\n";
            }
        }
        bucketOutput.close();
    }

    size_t bytes() { return n * sizeof(experimental::CbmStsDigi); }

};
