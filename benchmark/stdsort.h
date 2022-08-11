#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>

class stdsort_bench : public benchmark {

    const size_t n;
    std::ofstream file;
    experimental::CbmStsDigi* digis;
    experimental::CbmStsDigi* output;
    std::vector<float> executionTimeMs;

public:
    stdsort_bench(const experimental::CbmStsDigi* in_digis, const size_t in_n) : n(in_n), digis(new experimental::CbmStsDigi[in_n]), output(new experimental::CbmStsDigi[in_n]) {
        std::copy(in_digis, in_digis + n, digis);
    }

    ~stdsort_bench() {
        delete[] digis;
        delete[] output;
    }

    std::string name() { return "std::sort"; }

    void setup() {
        file.open("std_sort_output.csv", std::ios::out | std::ios::trunc);
    }

    void teardown() {
        check();

        file << "index,address,channel,time\n";

        for (int i = 0; i < n; i++) {
            file << i << "," << output[i].address << "," << output[i].channel << "," << output[i].time << "\n";
        }

        file.close();
    }

    void run() {
        // Copy for each run a fresh output original digi array.
        std::copy(digis, digis + n, output);

        auto started = std::chrono::high_resolution_clock::now();

        std::sort(output, output + n, [](const experimental::CbmStsDigi& a, const experimental::CbmStsDigi& b) {
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
            ok &= output[i].channel >= output[i - 1].channel;
            bool okThisRun = output[i].channel >= output[i - 1].channel;

            // Only check the time_j < time_j-1 within the same channels.
            if (output[i].channel == output[i - 1].channel) {
                ok &= output[i].time >= output[i - 1].time;
                okThisRun &= output[i].time >= output[i - 1].time;
            }

            if (!okThisRun) {
                std::cerr << "Error: " << "\n";
                std::cerr << "i-1: " << i - 1 << "(" << output[i - 1].address << ", " << output[i - 1].channel << ", " << output[i - 1].time << ")\n";
                std::cerr << "i:   " << i << "(" << output[i].address << ", " << output[i].channel << ", " << output[i].time << ")\n\n";
            }
        }

        if (ok) {
            std::cout << "Data is sorted!" << std::endl;
        } else {
            std::cerr << "Error: Data is not sorted!" << std::endl;
        }
    }

    size_t bytes() { return n * sizeof(experimental::CbmStsDigi); }

    std::vector<float> timings() { return executionTimeMs; }

};
