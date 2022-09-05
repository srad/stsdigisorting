#pragma once

#include "../datastructures.h"
#include "../common.h"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include <sqlite_orm/sqlite_orm.h>

namespace experimental {

    struct BenchmarkRun {
        int benchmarkId;
        std::string file;
        int n;
        std::string device;
        std::string timestamp;
    };

    struct BenchmarkEntry {
        int benchmarkEntryId;
        int benchmarkId;
        std::string name;
        float throughput;
        float median;
        int blockDimX;
        int itemsPerThread;
    };

    struct BenchmarkInfo {
        std::string name;
        int blockDimX;
        int itemsPerThread;
    };

    class benchmark {

    public:
        bool write_;
        bool check_;
        std::vector<float> timings_;

        benchmark(const bool in_write = false, const bool in_check = true) : write_(in_write), check_(in_check) {}
        virtual ~benchmark() {}

        virtual BenchmarkInfo info() = 0;
        virtual void setup() = 0;
        virtual void teardown() = 0;
        virtual void run() = 0;
        virtual size_t size() const = 0;

        virtual digi_t* output() = 0;

        virtual size_t bytes() const { return 0; }

        virtual std::vector<float> timings() { return timings_; }

        virtual void write() {
            create_dir("output");
            digi_t* sorted = output();

            // +------------------------------------------------------------------------------+
            // |                               Sorted output                                  |
            // +------------------------------------------------------------------------------+
            std::ofstream file;
            file.open("output/" + info().name + ".csv", std::ios::out | std::ios::trunc);
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
                    std::cout << info().name << " Error: " << "\n";
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
        benchmark_runner(const std::string in_subfolder, const std::string in_input_file) : subfolder(in_subfolder), input_file(in_input_file) {}

        void add(benchmark* b) { benchmarks.emplace_back(b); }

        inline auto init_storage(const std::string& path) {
            using namespace sqlite_orm;
            return make_storage(path,
                                make_table("BenchmarkRuns",
                                        make_column("BenchmarkId", &BenchmarkRun::benchmarkId, primary_key()),
                                        make_column("File", &BenchmarkRun::file),
                                        make_column("N", &BenchmarkRun::n),
                                        make_column("Device", &BenchmarkRun::device),
                                        make_column("Timestamp", &BenchmarkRun::timestamp)),

                                make_table("BenchmarkResults",
                                        make_column("BenchmarkEntryId", &BenchmarkEntry::benchmarkEntryId, primary_key()),
                                        make_column("BenchmarkId", &BenchmarkEntry::benchmarkId),
                                        make_column("Name", &BenchmarkEntry::name),
                                        make_column("ThroughputGbs", &BenchmarkEntry::throughput),
                                        make_column("MedianMs", &BenchmarkEntry::median),
                                        make_column("BlockDimX", &BenchmarkEntry::blockDimX),
                                        make_column("ItemsPerThread", &BenchmarkEntry::itemsPerThread),
                                        foreign_key(&BenchmarkEntry::benchmarkId).references(&BenchmarkRun::benchmarkId))
                                );
        }

        void run(int n) {
            auto storage = init_storage("benchmarks.sqlite");
            storage.sync_schema();

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
                    output << ",\"" << b.get()->info().name << "\"";
                    tp << ",\"" << b.get()->info().name << "\"";
                }
                run_benchmark(b.get(), n);
            }

            // +------------------------------------------------------------------------------+
            // |  Benchmark complete. Add to database.                                        |
            // +------------------------------------------------------------------------------+
            storage.transaction([&] () mutable {    
                BenchmarkRun run{-1, get_filename(input_file), (int)benchmarks[0].get()->size(), get_device(), storage.current_timestamp()};
                run.benchmarkId = storage.insert(run);
                for (auto& b: benchmarks) {
                            const BenchmarkInfo info = b.get()->info();
                            const auto ts = timings(b.get());

                            // +------------------------------------------------------------------------------+
                            // | Each sorting algorithm result                                                |
                            // +------------------------------------------------------------------------------+
                            BenchmarkEntry entry{-1, run.benchmarkId, info.name, get_throughput(b.get()), ts.median, info.blockDimX, info.itemsPerThread};
                            entry.benchmarkEntryId = storage.insert(entry);
                }
                return true;
            });

            // +------------------------------------------------------------------------------+
            // |  Print output and write to CSV.                                              |
            // +------------------------------------------------------------------------------+
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
                const BenchmarkInfo info = b.get()->info();
                const auto ts = timings(b.get());

                output << "," << ts.median;
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
        const std::string input_file;

        void run_benchmark(benchmark* b, const int r) {
            std::cout << "------------------------------------------------------------\nRunning benchmark '" << b->info().name << "'\n------------------------------------------------------------" << std::endl;
            b->setup();

            for (int i = 0; i < r + 1; i++) {
                b->run();
            }

            if (b->write_) { b->write(); }
            if (b->check_) { std::cout << "Checking " << b->info().name << "\n"; b->check(); }

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

            print_entry(b->info().name);

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