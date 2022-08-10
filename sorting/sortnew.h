#include "datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"

#include <iostream>
#include <random>
#include <vector>

template<typename Kernel>
class blocksort_bench : public benchmark {

    size_t n;
    const size_t elems_per_block = 32 * 32 * 200;
    size_t n_blocks;

    experimental::CbmStsDigi* digis;
    xpu::hd_buffer<experimental::CbmStsDigi> a;
    xpu::hd_buffer<experimental::CbmStsDigi> b;

public:
    xpu::hd_buffer<experimental::CbmStsDigi*> dst;

    blocksort_bench(experimental::CbmStsDigi* in_digis, size_t in_n) : digis(in_digis), n(in_n), n_blocks(in_n/elems_per_block) {}

    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        a = xpu::hd_buffer<experimental::CbmStsDigi>(n);
        b = xpu::hd_buffer<experimental::CbmStsDigi>(n);
        dst = xpu::hd_buffer<experimental::CbmStsDigi*>(n_blocks);

        std::copy(digis, digis + n, a.h());

        for (int i=0; i < 10; i++) {
            std::cout << a.h()[i].to_string() << "\n";
        }
        std::cout << "\n";
    }

    void teardown() {
        int k=0;
        for (int i=0; i < n_blocks; i++) {
            for (int j=0; j < elems_per_block; j++) {
                if (k++ == 10) {
                    goto out;
                }
                std::cout << dst.h()[i][j].to_string() << "\n";
            }
        }
        std::cout << "\n";

out:

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
        for (size_t i = 1; i < n; i++) {
            std::cout << "i: " << i << ", " << digis[i].to_string() << "\n";
            auto faa = (digis[i].channel <= digis[i-1].channel);
            ok &= faa;
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
