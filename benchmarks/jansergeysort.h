#pragma once

#include "../src/types.h"
#include "../src/datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"
#include <iostream>
#include <vector>
#include <chrono>

namespace experimental {

    template<typename Kernel>
    class jansergeysort_bench : public benchmark {

        const size_t n;
        const unsigned int blocksPerBucket;
        const std::string name;

        // Big difference here is that the digis are grouped in buckets and then bucket-wise sorted.
        bucket_t* bucket;

        CbmStsDigiInput* digis;
        xpu::hd_buffer<digi_t> buffDigis;
        xpu::hd_buffer<digi_t> buffOutput;

        xpu::hd_buffer<index_t> buffStartIndex;
        xpu::hd_buffer<index_t> buffEndIndex;

    public:
        jansergeysort_bench(const std::string in_name, const CbmStsDigiInput* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true, unsigned int in_block_per_bucket = 2) : n(in_n), digis(new CbmStsDigiInput[in_n]), name(in_name), blocksPerBucket(in_block_per_bucket), benchmark(in_write, in_check) {
            std::copy(in_digis, in_digis + in_n, digis);
            std::cout << "(" << info().name << ")" << " Block per bucket=" << blocksPerBucket << "\n";
        }

        ~jansergeysort_bench() {}

        BenchmarkInfo info() override {
            const auto kernel = std::string(xpu::get_name<Kernel>());
            return BenchmarkInfo{name, JanSergeySortBlockDimX, 0};
        }

        void setup() {
            buffDigis = xpu::hd_buffer<digi_t>(n);        
            buffOutput = xpu::hd_buffer<digi_t>(n);

            bucket = new bucket_t(digis, n);
            std::cout << "Buckets created." << "\n";

            buffStartIndex = xpu::hd_buffer<index_t>(bucket->size());
            buffEndIndex = xpu::hd_buffer<index_t>(bucket->size());

            std::copy(bucket->startIndex, bucket->startIndex + bucket->size(), buffStartIndex.h());
            std::copy(bucket->endIndex, bucket->endIndex + bucket->size(), buffEndIndex.h());

            std::copy(bucket->digis, bucket->digis + n, buffDigis.h());
        }

        void teardown() override {
            delete[] digis;
            delete bucket;
            buffStartIndex.reset();
            buffEndIndex.reset();
            buffDigis.reset();
            buffOutput.reset();
        }

        size_t size_n() const { return n; }

        void run() override {
            xpu::copy(buffDigis, xpu::host_to_device);

            xpu::copy(buffStartIndex, xpu::host_to_device);
            xpu::copy(buffEndIndex, xpu::host_to_device);

            // For now, one block per bucket.
            xpu::run_kernel<Kernel>(xpu::grid::n_blocks(bucket->size() * blocksPerBucket), n, buffDigis.d(), buffStartIndex.d(), buffEndIndex.d(), buffOutput.d());

            // Copy result back to host.
            xpu::copy(buffOutput, xpu::device_to_host);
        }

        std::vector<float> timings() override { return xpu::get_timing<Kernel>(); }

        size_t size() const { return n; }

        digi_t* output() override { return buffOutput.h(); }

        size_t bytes() const { return n * sizeof(digi_t); }

    };

}