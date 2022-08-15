#pragma once

#include <algorithm>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include "../common.h"

class benchmark {

public:
    const bool write_;
    const bool check_;

    benchmark(const bool in_write = false, const bool in_check = true) : write_(in_write), check_(in_check) {}
    virtual ~benchmark() {}

    virtual std::string name() = 0;
    virtual void setup() = 0;
    virtual void teardown() = 0;
    virtual void run() = 0;
    virtual size_t size_n() const = 0;
    virtual void write() { std::cout << "Benchmark->wirte() not implemented.\n"; }
    virtual void check() { std::cout << "Benchmark->check() not implemented.\n"; }

    virtual size_t bytes() { return 0; }

    virtual std::vector<float> timings() = 0;

};

struct bench_result {
    float min;
    float max;
    float median;
};

class benchmark_runner {

public:
    const std::string filename = "benchmark_results.csv";
    void add(benchmark* b) { benchmarks.emplace_back(b); }

    void run(int n) {
        for (auto& b: benchmarks) {
            run_benchmark(b.get(), n);
        }

        results_csv.open(filename, std::ios::out | std::ios::app);

        // Headers
        if (experimental::file_empty(filename)) {
            results_csv << "n";
            for (int i=0; i < benchmarks.size(); i++) {
                results_csv << "," << benchmarks[i].get()->name();
            }
            results_csv << "\n";
        }

        print_entry("Benchmark");
        print_entry("Min");
        print_entry("Max");
        print_entry("Median");
        std::cout << std::endl;

        // Rows
        results_csv << benchmarks[0].get()->size_n();
        for (auto& b: benchmarks) {
            const auto res = results(b.get());
            results_csv << "," << res.median;
            //print_results(b.get());
        }
        results_csv << "\n";
        results_csv.close();
    }

private:
    std::ofstream results_csv;
    std::vector <std::unique_ptr<benchmark>> benchmarks;

    void run_benchmark(benchmark* b, const int r) {
        std::cout << "Running benchmark '" << b->name() << "'" << std::endl;
        b->setup();

        for (int i = 0; i < r + 1; i++) {
            b->run();
        }

        if (b->write_) { b->write(); }
        if (b->check_) { std::cout << "Checking " << b->name() << "\n"; b->check(); }

        b->teardown();
    }

    bench_result results(benchmark* b) const {
        std::vector<float> timings = b->timings();

        timings.erase(timings.begin()); // discard warmup run
        std::sort(timings.begin(), timings.end());

        print_entry(b->name());

        float min = timings.front();
        float max = timings.back();
        float median = timings[timings.size() / 2];

        return bench_result{min, max, median};
    }

    void print_results(benchmark* b) {
        const auto result = results(b);
        size_t bytes = b->bytes();

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);

        ss << result.min << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, result.min) << "GB/s)";
        }
        print_entry(ss.str());
        ss.str("");

        ss << result.max << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, result.max) << "GB/s)";
        }
        print_entry(ss.str());
        ss.str("");

        ss << result.median << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, result.median) << "GB/s)";
        }
        print_entry(ss.str());
        ss.str("");
        std::cout << std::endl;
    }

    void print_entry(std::string entry) const {
        std::cout << std::left << std::setw(25) << std::setfill(' ') << entry;
    }

    float gb_s(size_t bytes, float ms) const {
        return (bytes / (1000.f * 1000.f * 1000.f)) / (ms / 1000.f);
    }

};
