#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"

#include <iostream>
#include <vector>

template<typename Kernel>
class blocksort_bench : public benchmark {

    const size_t n;
    const size_t elems_per_block;
    const size_t n_blocks = 64;

    // Write the sorted result to CSV file.
    std::ofstream output;

    experimental::CbmStsDigi* digis;
    experimental::CbmStsDigi* sorted;

    xpu::hd_buffer <experimental::CbmStsDigi> buffInput;
    xpu::d_buffer <experimental::CbmStsDigi> buffTemp;
    xpu::hd_buffer <experimental::CbmStsDigi*> buffOutput;

public:
    blocksort_bench(const experimental::CbmStsDigi* in_digis, const size_t in_n) : n(in_n), elems_per_block(ceil((double)in_n/n_blocks)), sorted(new experimental::CbmStsDigi[in_n]) {
        // Create an internal copy of the digis.
        std::copy(in_digis, in_digis + in_n, digis);
        printf("n: %lu, n_blocks: %lu, elems_per_block: %lu\n", n, n_blocks, elems_per_block);
    }

    ~blocksort_bench() {
        delete[] digis;
        delete[] sorted;
    }

    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        buffInput = xpu::hd_buffer<experimental::CbmStsDigi>(n);
        buffTemp = xpu::d_buffer<experimental::CbmStsDigi>(n);
        buffOutput = xpu::hd_buffer<experimental::CbmStsDigi*>(n_blocks);

        output.open("block_sort_output.csv", std::ios::out | std::ios::trunc);
        printf("block_sort: n=%lu, n_blocks=%lu, elems_per_block=%lu\n", n, n_blocks, elems_per_block);
    }

    void run() {
        std::copy(digis, digis + n, buffInput.h());
        xpu::copy(buffInput, xpu::host_to_device);

        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(n_blocks), buffInput.d(), buffTemp.d(), buffOutput.d(), n);

        // Sorting completed, result written block-wise into this array.
        xpu::copy(buffOutput, xpu::device_to_host);

        /*
        std::copy(buffInput.h(), buffOutput.h()[0]);

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
        //check();
        /*
        output << "index,address,channel,time\n";

        int k = 0;
        for (int i = 0; i < n_blocks; i++) {
            for (int j = 0; j < elems_per_block && k < n; j++) {
                output << k++ << "," << buffOutput.h()[i][j].address << "," << buffOutput.h()[i][j].channel << "," << buffOutput.h()[i][j].time << "\n";
            }
        }
        */
        output.close();
        buffInput.reset();
        buffTemp.reset();
        buffOutput.reset();
    }

    void check() {
        // Check if data is sorted.
        bool ok = true;

        size_t k=0;
        for (size_t i = 0; i < n; i++) {
            bool okThisRun = true;
            const auto& curr = sorted[i];
            const auto& prev = sorted[i - 1];

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
