#pragma once

#include "../src/types.h"
#include "../src/datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"
#include <iostream>
#include <vector>

namespace experimental {

    template<typename PartitionKernel>
    class partition_bench : public benchmark {

        const size_t n;
        const std::string name;

        CbmStsDigiInput* digis;

        xpu::hd_buffer<digi_t> hd_input;
        xpu::hd_buffer<digi_t> hd_output;

        xpu::hd_buffer<index_t> startIndex;
        xpu::hd_buffer<index_t> endIndex;

        // Big difference here is that the digis are grouped in buckets and then bucket-wise sorted.
        CbmStsDigiBucket* bucket;

    public:
        partition_bench(const std::string in_name, const CbmStsDigiInput* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true) : n(in_n), digis(new CbmStsDigiInput[in_n]), name(in_name), benchmark(in_write, in_check) {
            std::copy(in_digis, in_digis + in_n, digis);
        }

        ~partition_bench() {}

        BenchmarkInfo info() override {
            return BenchmarkInfo{name, PartitionBlockDimX, 0};
        }

        void setup() {
            hd_input = xpu::hd_buffer<digi_t>(n);        
            hd_output = xpu::hd_buffer<digi_t>(n);

            bucket = new CbmStsDigiBucket(digis, n);
            std::cout << "Parition CbmStsDigiBucket created." << "\n";

            startIndex = xpu::hd_buffer<index_t>(bucket->size());
            endIndex = xpu::hd_buffer<index_t>(bucket->size());

            // Copy data to buffers.
            std::copy(bucket->startIndex, bucket->startIndex + bucket->size(), startIndex.h());
            std::copy(bucket->endIndex, bucket->endIndex + bucket->size(), endIndex.h());
            std::copy(bucket->digis, bucket->digis + n, hd_input.h());
        }

        void teardown() override {
            delete[] digis;
            delete bucket;
            startIndex.reset();
            endIndex.reset();
            hd_input.reset();
            hd_output.reset();
        }

        size_t size_n() const { return n; }

        void run() override {
            xpu::copy(hd_input, xpu::host_to_device);

            xpu::copy(startIndex, xpu::host_to_device);
            xpu::copy(endIndex, xpu::host_to_device);

            // For now, one block per bucket.
            // digi_t* input, const index_t* startIndex, const index_t* endIndex, digi_t* output, const size_t n
            xpu::run_kernel<PartitionKernel>(xpu::grid::n_blocks(bucket->size()), hd_input.d(), startIndex.d(), endIndex.d(), hd_output.d(), n);

            // Copy result back to host.
            xpu::copy(hd_output, xpu::device_to_host);
        }

        std::vector<float> timings() override { return xpu::get_timing<PartitionKernel>(); }

        size_t size() const { return n; }

        digi_t* output() override { return hd_output.h(); }

        size_t bytes() const { return n * sizeof(digi_t); }

    };

}