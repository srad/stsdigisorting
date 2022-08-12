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

    // Write the sorted result to CSV file.
    std::ofstream output;

    // This is an internal copy of the unsorted digis
    // which is copied to run the sort benchmark repeatedly.
    experimental::CbmStsDigi* digis;

    // All device allocations.
    experimental::CbmStsDigi* inputD;
    experimental::CbmStsDigi* bufD;
    experimental::CbmStsDigi** outD;

    experimental::CbmStsDigi* itemsH;

public:
    blocksort_bench(const experimental::CbmStsDigi* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true) : n(in_n), itemsH(new experimental::CbmStsDigi[in_n]), digis(new experimental::CbmStsDigi[in_n]), benchmark(in_write, in_check) {
        // Create an internal copy of the digis.
        std::copy(in_digis, in_digis + in_n, digis);
        elems_per_block = n / n_blocks;
    }

    ~blocksort_bench() {}

    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        std::cout << "n=" << n << ", n_blocks=" << n_blocks << ", elems_per_block=" << elems_per_block << "\n";
        inputD = xpu::device_malloc<experimental::CbmStsDigi>(n);
        bufD = xpu::device_malloc<experimental::CbmStsDigi>(n);
        outD = xpu::device_malloc<experimental::CbmStsDigi*>(1);
    }

    void run() {
        // Copy a fresh copy over itemsH to sort again.
        std::copy(digis, digis + n, itemsH);

        // Copy data from host to GPU.
        xpu::copy(inputD, itemsH, n);

        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(n_blocks), inputD, bufD, outD, n);

        // Get the buffer that contains the sorted data.
        experimental::CbmStsDigi* outH = nullptr;
        xpu::copy(&outH, outD, 1);

        // Copy to back host.
        xpu::copy(itemsH, outH, n);

        /*
        size_t k=0;
        // Copy block-wise the sorted results to the flat array "sorted".
        for (int i=0; i < n_blocks; i++) {
            xpu::copy(buffSortedBlock.h(), buffOutput.h()[0]);
            for(int j=0; j < elems_per_block && k < n; j++, k++) {
                sorted[k] = buffSortedBlock.h()[j];
            }
            //std::copy(buffOutput.h(), buffOutput.h()[i] + elems_per_block, sorted + (i * elems_per_block));
        }
        */
    }

    void teardown() {
        delete[] digis;
        delete[] itemsH;
        xpu::free(inputD);
        xpu::free(bufD);
        xpu::free(outD);
        /*

    bool ok = true;
    for (size_t block = 0; block < n_blocks; block++) {
        size_t offset = block * elems_per_block;
        for (size_t i = 1; i < elems_per_block; i++) {
            auto faa = (itemsH[offset+i-1].key <= itemsH[offset+i].key);
            ok &= faa;
        }
    }

        output.open("block_sort_output.csv", std::ios::out | std::ios::trunc);
        output << "index,address,channel,time\n";

        int k = 0;
        for (int i = 0; i < n_blocks; i++) {
            for (int j = 0; j < elems_per_block && k < n; j++) {
                output << k++ << "," << buffOutput.h()[i][j].address << "," << buffOutput.h()[i][j].channel << "," << buffOutput.h()[i][j].time << "\n";
            }
        }
        */
    }

    void check_skip_for_now() {
        // Check if data is sorted.
        bool ok = true;

        size_t k=0;
        for (size_t i = 0; i < n; i++) {
            bool okThisRun = true;
            const auto& curr = itemsH[i];
            const auto& prev = itemsH[i - 1];

            // Only check the time_j < time_j-1 within the same channels.
            if (curr.address == prev.address) {
                ok &= curr.channel >= prev.channel;
                okThisRun &= curr.channel >= prev.channel;
            }
            if (curr.channel == prev.channel) {
                ok &= curr.time >= prev.time;
                okThisRun &= curr.time >= prev.time;
            }

            if (!okThisRun) {
                std::cout << "BlockSort Error: " << "\n";
                printf("(%lu/%lu) (%d, %d, %d)\n", i - 1, n, prev.address, prev.channel, prev.time);
                printf("(%lu/%lu) (%d, %d, %d)\n", i,     n, curr.address, curr.channel, curr.time);
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
