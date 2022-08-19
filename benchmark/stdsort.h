#pragma once

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
    experimental::CbmStsDigiInput* digis;
    experimental::CbmStsDigi* sorted;

public:
    stdsort_bench(const experimental::CbmStsDigiInput* in_digis, const size_t in_n, const bool write = false, const bool check = true) : n(in_n), digis(new experimental::CbmStsDigiInput[in_n]), sorted(new experimental::CbmStsDigi[in_n]), benchmark(write, check) {
        std::copy(in_digis, in_digis + n, digis);
    }

    ~stdsort_bench() {
    }

    std::string name() { return "std_sort"; }

    void setup() {}

    void teardown() {
        delete[] digis;
        delete[] sorted;
    }

    void run() {
        // Copy for each run a fresh output original digi array.
        for (int i=0; i < n; i++) {
            sorted[i] = experimental::CbmStsDigi(digis[i].channel, digis[i].time);
        }

        auto started = std::chrono::high_resolution_clock::now();

        std::sort(sorted, sorted + n, [](const experimental::CbmStsDigi& a, const experimental::CbmStsDigi& b) {
            return (((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time)) < (((unsigned long int) b.channel) << 32 | (unsigned long int) (b.time));
        });

        auto done = std::chrono::high_resolution_clock::now();
        timings_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count());
    }

    size_t size() override { return n; }

    experimental::CbmStsDigi* output() override { return sorted; }

    size_t bytes() { return n * sizeof(experimental::CbmStsDigi); }

};
