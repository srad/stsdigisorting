#pragma once

#include "../types.h"
#include "../datastructures.h"

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

        // Big difference here is that the digis are grouped in buckets and then bucket-wise sorted.
        bucket_t* bucket;

        CbmStsDigiInput* digis;
        xpu::hd_buffer<digi_t> buffDigis;
        xpu::hd_buffer<digi_t> buffOutput;

        xpu::hd_buffer<index_t> buffStartIndex;
        xpu::hd_buffer<index_t> buffEndIndex;

        xpu::hd_buffer<index_t> channelSplitIndex;

    public:
        jansergeysort_bench(const CbmStsDigiInput* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true, unsigned int in_block_per_bucket = 2) : n(in_n), digis(new CbmStsDigiInput[in_n]), blocksPerBucket(in_block_per_bucket), benchmark(in_write, in_check) {
            std::copy(in_digis, in_digis + in_n, digis);
            std::cout << "(" << name() << ")" << " Block per bucket=" << blocksPerBucket << "\n";
        }

        ~jansergeysort_bench() {}

        std::string name() {
            const auto kernel = std::string(xpu::get_name<Kernel>());
            const auto str = kernel.substr(kernel.find("::") + 2, kernel.length());
            return  str + "(" + std::to_string(JanSergeySortTPB) + "," + get_device() + ")";
        }

        void setup() {
            buffDigis = xpu::hd_buffer<digi_t>(n);        
            buffOutput = xpu::hd_buffer<digi_t>(n);

            bucket = new bucket_t(digis, n);
            std::cout << "Buckets created." << "\n";

            buffStartIndex = xpu::hd_buffer<index_t>(bucket->size());
            buffEndIndex = xpu::hd_buffer<index_t>(bucket->size());

            channelSplitIndex = xpu::hd_buffer<index_t>(bucket->size());

            std::copy(bucket->startIndex, bucket->startIndex + bucket->size(), buffStartIndex.h());
            std::copy(bucket->endIndex, bucket->endIndex + bucket->size(), buffEndIndex.h());

            std::copy(bucket->channelSplitIndex, bucket->channelSplitIndex + bucket->size(), channelSplitIndex.h());

            std::copy(bucket->digis, bucket->digis + n, buffDigis.h());
        }

        void teardown() {
            delete[] digis;
            delete bucket;
            buffStartIndex.reset();
            buffEndIndex.reset();
            buffDigis.reset();
            buffOutput.reset();
            channelSplitIndex.reset();
        }

        size_t size_n() const { return n; }

        void run() {
            xpu::copy(buffDigis, xpu::host_to_device);

            xpu::copy(buffStartIndex, xpu::host_to_device);
            xpu::copy(buffEndIndex, xpu::host_to_device);

            xpu::copy(channelSplitIndex, xpu::host_to_device);

            // For now, one block per bucket.
            xpu::run_kernel<Kernel>(xpu::grid::n_blocks(bucket->size() * blocksPerBucket), n, buffDigis.d(), buffStartIndex.d(), buffEndIndex.d(), buffOutput.d(), channelSplitIndex.d());

            // Copy result back to host.
            xpu::copy(buffOutput, xpu::device_to_host);
        }

        std::vector<float> timings() override { return xpu::get_timing<Kernel>(); }

        size_t size() override { return n; }

        digi_t* output() override { return buffOutput.h(); }

        void write() override {
            benchmark::write();
            
            // +------------------------------------------------------------------------------+
            // |                               Bucket data                                    |
            // +------------------------------------------------------------------------------+
            /*
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
            */
        }

        size_t bytes() { return n * sizeof(digi_t); }

    };

}