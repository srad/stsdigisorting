#pragma once

#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"
#include <iostream>
#include <vector>

template<typename Kernel>
class jansergeysort_bench : public benchmark {

    const size_t n;

    // Big difference here is that the digis are grouped in buckets and then bucket-wise sorted.
    experimental::CbmStsDigiBucket* bucket;

    // Write the sorted result to CSV file.
    std::ofstream output;
    std::ofstream bucketOutput;

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

    void run() {
        xpu::copy(buffDigis, xpu::host_to_device);
        xpu::copy(buffStartIndex, xpu::host_to_device);
        xpu::copy(buffEndIndex, xpu::host_to_device);

        // For now, one block per bucket.
        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(bucket->size()), n, buffDigis.d(), buffStartIndex.d(), buffEndIndex.d(), buffOutput.d());

        // Copy result back to host.
        xpu::copy(buffOutput, xpu::device_to_host);
    }

    void write() override {
        bucketOutput.open("bucket.csv", std::ios::out | std::ios::trunc);
        bucketOutput << "bucket,index,address,channel,time\n";

        for(int i=0; i < bucket->size(); i++) {
            const auto start = bucket->startIndex[i];
            const auto end   = bucket->endIndex[i];
            for(int j=start; j <= end; j++) {
                bucketOutput << i << "," << j << "," << bucket->digis[j].address << "," << bucket->digis[j].channel << "," << bucket->digis[j].time  << "\n";
            }
        }

        output.open("jansergey_sort_output.csv", std::ios::out | std::ios::trunc);
        output << "index,address,channel,time\n";

        for (int i = 0; i < n; i++) {
            output << i << "," << buffOutput.h()[i].address << "," << buffOutput.h()[i].channel << "," << buffOutput.h()[i].time << "\n";
        }

        output.close();
        bucketOutput.close();
    }

    void check() override {
        // Check if data is sorted.
        bool ok = true;

        // Start at second element and compare to previous for all i.
        for (size_t i = 1; i < n; i++) {
            bool okThisRun = true;

            const auto& curr = buffOutput.h()[i];
            const auto& prev = buffOutput.h()[i - 1];

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
                std::cout << "\nJanSergeySort Error: " << "\n";
                printf("(%lu/%lu): (%d, %d, %d)\n", i-1, n, prev.address, prev.channel, prev.time);
                printf("(%lu/%lu): (%d, %d, %d)\n", i, n, curr.address, curr.channel, curr.time);
            }
        }

        if (ok) {
            std::cout << "Data is sorted!" << std::endl;
        } else {
            std::cout << "Error: Data is not sorted!" << std::endl;
        }
    }

    size_t bytes() { return n * sizeof(experimental::CbmStsDigi); }

    std::vector<float> timings() { return xpu::get_timing<Kernel>(); }

};
