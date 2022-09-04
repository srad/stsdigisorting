#pragma once

#include "../datastructures.h"
#include <algorithm>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include "../common.h"

namespace experimental {

    class benchmark {

    public:
        bool write_;
        bool check_;
        std::vector<float> timings_;

        benchmark(const bool in_write = false, const bool in_check = true) : write_(in_write), check_(in_check) {}
        virtual ~benchmark() {}

        virtual std::string name() = 0;
        virtual void setup() = 0;
        virtual void teardown() = 0;
        virtual void run() = 0;
        virtual size_t size() = 0;
        virtual digi_t* output() = 0;

        virtual size_t bytes() { return 0; }

        virtual std::vector<float> timings() { return timings_; }

        virtual void write() {
            create_dir("output");
            digi_t* sorted = output();

            // +------------------------------------------------------------------------------+
            // |                               Sorted output                                  |
            // +------------------------------------------------------------------------------+
            std::ofstream file;
            file.open("output/" + name() + ".csv", std::ios::out | std::ios::trunc);
            file << "index," + digi_t::csv_headers()  + "\n";

            for (int i = 0; i < size(); i++) {
                file << i << "," << sorted[i].to_csv() << "\n";
            }

            file.close();
        }

        virtual void check() {
            const digi_t* sorted = output();

            // Check if data is sorted.
            bool ok = true;
            unsigned int errorCount = 0;

            // Start at second element and compare to previous for all i.
            for (size_t i = 1; i < size(); i++) {
                bool okThisRun = true;

                const auto& curr = sorted[i];
                const auto& prev = sorted[i - 1];

                // Within the same address range, channel numbers increase.
                if (curr.channel == prev.channel) {
                    ok &= curr.time >= prev.time;
                    okThisRun &= curr.time >= prev.time;
                }

                if (!okThisRun) {
                    errorCount++;
                    std::cout << name() << " Error: " << "\n";
                    printf("(%lu/%lu): (%d, %d)\n", i-1, size(), prev.channel, prev.time);
                    printf("(%lu/%lu): (%d, %d)\n\n", i, size(), curr.channel, curr.time);
                }
            }

            if (ok) {
                std::cout << "Data is sorted!" << std::endl;
            } else {
                std::cout << "Error: Data is not sorted!" << "\n";
                std::cout << "Error count: " << errorCount << "\n";
            }
        }

    };

    class benchmark_runner {

    public:
        benchmark_runner(const std::string in_subfolder = "") : subfolder(in_subfolder) {}

        void add(benchmark* b) { benchmarks.emplace_back(b); }

        void run(int n) {
            const std::string filename = (subfolder != "") ? subfolder + "/benchmark_results.csv" : "benchmark_results.csv";
            const std::string tp_filename = (subfolder != "") ? subfolder + "/benchmark_tp.csv" : "benchmark_tp.csv";

            const bool exists = file_exists(filename);

            std::ofstream output;
            output.open(filename, std::ios::out | std::ios_base::app);

            std::ofstream tp;
            tp.open(tp_filename, std::ios::out | std::ios_base::app);

            // Header

            // n col.
            if (!exists) { output << "\"n\""; tp << "\"n\""; }
            std::cout << "Writing benchmark results ..\n";

            // Benchmark names as cols.
            for (auto& b: benchmarks) {
                if (!exists) {
                    output << ",\"" << b.get()->name() << "\"";
                    tp << ",\"" << b.get()->name() << "\"";
                }
                run_benchmark(b.get(), n);
            }
            // Line end.
            if (!exists) { output << "\n"; tp << "\n"; }

            print_entry("Benchmark");
            print_entry("Min");
            print_entry("Max");
            print_entry("Median");
            std::cout << std::endl;

            output << benchmarks[0].get()->size();
            tp << benchmarks[0].get()->size();
            for (auto& b: benchmarks) {
                output << "," << timings(b.get()).median;
                tp << "," << get_throughput(b.get());
                print_results(b.get());
            }
            output << "\n";
            tp << "\n";

            output.close();
            tp.close();
        }

    private:
        std::vector <std::unique_ptr<benchmark>> benchmarks;
        const std::string subfolder;

        void run_benchmark(benchmark* b, const int r) {
            std::cout << "------------------------------------------------------------\nRunning benchmark '" << b->name() << "'\n------------------------------------------------------------" << std::endl;
            b->setup();

            for (int i = 0; i < r + 1; i++) {
                b->run();
            }

            if (b->write_) { b->write(); }
            if (b->check_) { std::cout << "Checking " << b->name() << "\n"; b->check(); }

            b->teardown();
            std::cout << "\n";
        }

        struct timing_results {
            float min;
            float max;
            float median;
        };

        timing_results timings(benchmark* b) {
            std::vector<float> timings = b->timings();

            timings.erase(timings.begin()); // discard warmup run
            std::sort(timings.begin(), timings.end());

            float min = timings.front();
            float max = timings.back();
            float median = timings[timings.size() / 2];

            return timing_results{min, max, median};
        }

        float get_throughput(benchmark* b) {
            const auto times = timings(b);
            const size_t bytes = b->bytes();
            return gb_s(bytes, times.median);
        }

        void print_results(benchmark* b) {
            const auto times = timings(b);
            const size_t bytes = b->bytes();

            print_entry(b->name());

            std::stringstream ss;
            ss << std::fixed << std::setprecision(5);

            ss << times.min << "ms";
            if (bytes > 0) {
                ss << " (" << gb_s(bytes, times.min) << " GB/s) ";
            }
            print_entry(ss.str());
            ss.str("");

            ss << times.max << "ms";
            if (bytes > 0) {
                ss << " (" << gb_s(bytes, times.max) << " GB/s) ";
            }
            print_entry(ss.str());
            ss.str("");

            ss << times.median << "ms";
            if (bytes > 0) {
                ss << " (" << gb_s(bytes, times.median) << " GB/s) ";
            }
            print_entry(ss.str());
            ss.str("");
            std::cout << std::endl;
        }

        void print_entry(std::string entry) const {
            std::cout << std::left << std::setw(30) << std::setfill(' ') << entry;
        }

        float gb_s(size_t bytes, float ms) const {
            return (bytes / (1000.f * 1000.f * 1000.f)) / (ms / 1000.f);
        }

    };

}