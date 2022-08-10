#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "../benchmark/benchmark.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>

class stdsort_bench : public benchmark {

    size_t n;
    std::ofstream output;
    experimental::CbmStsDigi* digis;
    std::vector<float> executionTimeMs;

public:
    stdsort_bench(const experimental::CbmStsDigi* in_digis, const size_t in_n) : n(in_n), digis(new experimental::CbmStsDigi[in_n]) {
        std::copy(in_digis, in_digis + n, digis);
    }

    ~stdsort_bench() {
        delete[] digis;
    }

    std::string name() { return "std::sort"; }

    void setup() {
        output.open("std_sort_output.csv", std::ios::out | std::ios::trunc);
    }

    void teardown() {
        check();

        output << "index,address,channel,time\n";

        for (int i = 0; i < n; i++) {
            output << i << "," << digis[i].address << "," << digis[i].channel << "," << digis[i].time << "\n";
        }

        output.close();
    }

    void run() {
        auto started = std::chrono::high_resolution_clock::now();
        //Not available on Linux yet.
        //std::sort(std::execution::par_unseq, digis, digis + n);
        std::sort(digis, digis + n, [](const experimental::CbmStsDigi& a, const experimental::CbmStsDigi& b) {
            return (((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time)) < (((unsigned long int) b.channel) << 32 | (unsigned long int) (b.time));
        });
        auto done = std::chrono::high_resolution_clock::now();
        executionTimeMs.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count());
    }

    void check() {
        // Check if data is sorted.
        bool ok = true;

        // Start at second digi.
        for (int i = 1; i < n; i++) {
            ok &= digis[i].channel >= digis[i - 1].channel;

            // Only check the time_j < time_j-1 within the same channels.
            if (digis[i].channel == digis[i - 1].channel) {
                auto faa = (digis[i].time >= digis[i - 1].time);
                ok &= faa;
            }

            if (!ok) {
                std::cout << "Error: " << "\n";
                printf("i: %d: (%d, %d, %d)\n", i, digis[i - 1].address, digis[i - 1].channel, digis[i - 1].time);
                printf("i: %d: (%d, %d, %d)\n", i, digis[i].address, digis[i].channel, digis[i].time);
                return;
            }
        }

        if (ok) {
            std::cout << "Data is sorted!" << std::endl;
        } else {
            std::cout << "Error: Data is not sorted!" << std::endl;
        }
    }

    size_t bytes() { return n * sizeof(experimental::CbmStsDigi); }

    std::vector<float> timings() { return executionTimeMs; }

};
