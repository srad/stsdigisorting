#pragma once

#include "../src/types.h"
#include "../src/datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "benchmark.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <thread>

namespace experimental {

    enum SortMode { seq, par };

    class stdsort_bench : public benchmark {
  
        const SortMode mode;
        const size_t n;
        CbmStsDigiInput* digis;
        digi_t* output_;
        bucket_t* bucket;

    std::string get_mode() const {
        switch(mode) {
            case SortMode::seq: return "::seq";
            case SortMode::par: return "::par";
        }
    }

    public:
        stdsort_bench(const CbmStsDigiInput* in_digis, const size_t in_n, const SortMode in_mode, const bool in_write = false, const bool in_check = true) : n(in_n), mode(in_mode), digis(new CbmStsDigiInput[in_n]), output_(new digi_t[in_n]), benchmark(in_write, in_check) {
            std::copy(in_digis, in_digis + n, digis);
        }

        ~stdsort_bench() {}

        BenchmarkInfo info() override {
            return BenchmarkInfo{"std::sort" + get_mode(), 0, 0};
        }

        void setup() override {
            bucket = new bucket_t(digis, n);
            std::cout << "Buckets created." << "\n";
        }

        void teardown() override {
            delete[] digis;
            delete[] output_;
        }

        static void sortBucket(digi_t* startSegment, digi_t* endSegment) {
            std::sort(startSegment, endSegment, [](const digi_t& a, const digi_t& b) {
                return (((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time)) < (((unsigned long int) b.channel) << 32 | (unsigned long int) (b.time));
            });
        }

        void run() override {
            // Copy for each run a fresh output original digi array.
            for (int i=0; i < n; i++) {
                output_[i] = digi_t(digis[i].address, digis[i].channel, digis[i].time);
            }

            // Create a fresh copy, in-place sorting.
            std::copy(bucket->digis, bucket->digis + n, output_);

            std::chrono::time_point<std::chrono::high_resolution_clock> started;

            switch(mode) {
                case SortMode::seq:
                    started = std::chrono::high_resolution_clock::now();

                    for (int i = 0; i < bucket->size(); i++) {
                        std::sort(output_ + bucket->begin(i), output_ + bucket->end(i), [](const digi_t& a, const digi_t& b) {
                            return (((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time)) < (((unsigned long int) b.channel) << 32 | (unsigned long int) (b.time));
                        });
                    }
                    break;
                case SortMode::par:
                    // Parallel sort each bucket.
                    std::vector<std::thread> threads(bucket->size());

                    started = std::chrono::high_resolution_clock::now();

                    for (int i = 0; i < bucket->size(); i++) {
                        threads[i] = std::thread(sortBucket, output_ + bucket->begin(i), output_ + bucket->end(i));
                    }
                    for (auto& th : threads) {
                        th.join();
                    }
                    break;
            }

            auto done = std::chrono::high_resolution_clock::now();
            timings_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count());
        }

        size_t size() const { return n; }

        digi_t* output() override { return output_; }

        size_t bytes() const { return n * sizeof(digi_t); }

    }; // class

} // namespace