#pragma once

#include <algorithm>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <sstream>
#include <vector>

class benchmark {

public:
    bool write_;
    bool check_;

    benchmark(const bool in_write = false, const bool in_check = true) : write_(in_write), check_(in_check) {}
    virtual ~benchmark() {}

    virtual std::string name() = 0;
    virtual void setup() = 0;
    virtual void teardown() = 0;
    virtual void run() = 0;
    virtual void write() { std::cout << "Benchmark->wirte() not implemented.\n"; }
    virtual void check() { std::cout << "Benchmark->check() not implemented.\n"; }

    virtual size_t bytes() { return 0; }

    virtual std::vector<float> timings() = 0;

};

class benchmark_runner {

public:
    void add(benchmark* b) { benchmarks.emplace_back(b); }

    void run(int n) {
        for (auto& b: benchmarks) {
            run_benchmark(b.get(), n);
        }

        print_entry("Benchmark");
        print_entry("Min");
        print_entry("Max");
        print_entry("Median");
        std::cout << std::endl;
        for (auto& b: benchmarks) {
            print_results(b.get());
        }
    }

private:
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

    void print_results(benchmark* b) {
        std::vector<float> timings = b->timings();
        size_t bytes = b->bytes();

        timings.erase(timings.begin()); // discard warmup run
        std::sort(timings.begin(), timings.end());

        print_entry(b->name());

        float min = timings.front();
        float max = timings.back();
        float median = timings[timings.size() / 2];

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);

        ss << min << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, min) << "GB/s)";
        }
        print_entry(ss.str());
        ss.str("");

        ss << max << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, max) << "GB/s)";
        }
        print_entry(ss.str());
        ss.str("");

        ss << median << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, median) << "GB/s)";
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
