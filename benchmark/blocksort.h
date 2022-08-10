#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "../benchmark/benchmark.h"

#include <iostream>
#include <random>
#include <vector>

template<typename Kernel>
class blocksort_bench : public benchmark {

    size_t n;
    const size_t elems_per_block = 32 * 32 * 200;
    size_t n_blocks;

    // Write the sorted result to CSV file.
    std::ofstream output;

    experimental::CbmStsDigi* digis;
    xpu::hd_buffer<experimental::CbmStsDigi> a;
    xpu::hd_buffer<experimental::CbmStsDigi> b;
    xpu::hd_buffer<experimental::CbmStsDigi*> dst;

public:
    blocksort_bench(experimental::CbmStsDigi* in_digis, size_t in_n) : digis(in_digis), n(in_n), n_blocks(in_n/elems_per_block) {}

    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        a = xpu::hd_buffer<experimental::CbmStsDigi>(n);
        b = xpu::hd_buffer<experimental::CbmStsDigi>(n);
        dst = xpu::hd_buffer<experimental::CbmStsDigi*>(n_blocks);

        std::copy(digis, digis + n, a.h());
        output.open("block_sort_output.csv", std::ios::out | std::ios::trunc);
        printf("block_sort: n=%llu, n_blocks=%llu, elems_per_block=%llu\n", n, n_blocks, elems_per_block);
    }

    void teardown() {
        check();

	output << "index,address,channel,time\n";

        int k=0;
        for (int i=0; i < n_blocks; i++) {
            for (int j=0; j < elems_per_block; j++) {
		output << k++ << "," << dst.h()[i][j].address << "," << dst.h()[i][j].channel << "," << dst.h()[i][j].time << "\n";
            }
        }

        output.close();
        a.reset();
        b.reset();
        dst.reset();
    }

    void run() {
        xpu::copy(a, xpu::host_to_device);
        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(n_blocks), a.d(), b.d(), dst.d(), elems_per_block);
        xpu::copy(dst, xpu::device_to_host);
    }

    void check() {
        // Check if data is sorted.
        bool ok = true;

        int k=0;
        for (int i=0; i < n_blocks; i++) {
            // Offset 1 and compare the i-th element with the i-1-th element.
            for (int j=1; j < elems_per_block && k < n; j++, k++) {
                bool okThisRun = true;

                // Digi i has always a >= bigger channel number after sorting than digi i-1.
                ok &= dst.h()[i][j].channel >= dst.h()[i][j-1].channel;
                okThisRun &= dst.h()[i][j].channel >= dst.h()[i][j-1].channel;

                // Only check the time_j < time_j-1 within the same channels.
                if (dst.h()[i][j].channel == dst.h()[i][j-1].channel) {
                    auto faa = (dst.h()[i][j].time >= dst.h()[i][j-1].time);
                    ok &= faa;
                    okThisRun &= (dst.h()[i][j].time >= dst.h()[i][j-1].time);
                }

                if (!okThisRun) {
                    std::cout << "Error: " << "\n";
                    printf("k: %d/%d, i: %d, j-1: %d: (%d, %d, %d)\n", k, n, i, j-1, dst.h()[i][j-1].address, dst.h()[i][j-1].channel, dst.h()[i][j-1].time);
                    printf("k: %d/%d, i: %d, j:   %d: (%d, %d, %d)\n", k, n, i, j, dst.h()[i][j].address, dst.h()[i][j].channel, dst.h()[i][j].time);
                }
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
