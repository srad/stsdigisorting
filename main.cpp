#include <iostream>
#include <cstring>
#include <cstdlib>
#include "common.h"

#include "benchmark/blocksort.h"
#include "benchmark/stdsort.h"
#include "benchmark/jansergeysort.h"

#include "sorting/BlockSort.h"
#include "sorting/JanSergeySort.h"
#include "sorting/JanSergeySortSingleBlock.h"
#include "sorting/JanSergeySortSimpleSum.h"
#include "sorting/JanSergeySortParInsert.h"

int main(int argc, char** argv) {
    try {
        // Command line params.
        std::string input;
        std::string output;
        unsigned int repeat = 1;
        unsigned int max_n = 0;
        bool writeOutput = false;
        bool checkResult = false;
        std::string benchmark_subfolder = "";

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0) {
                input = argv[i + 1];
                std::cout << "Input: " << input << "\n";
            } else if (strcmp(argv[i], "-o") == 0) {
                output = argv[i + 1];
                std::cout << "Output: " << output << "\n";
            } else if (strcmp(argv[i], "-c") == 0) {
                checkResult = true;
                std::cout << "Checking result.\n";
            } else if (strcmp(argv[i], "-w") == 0) {
                writeOutput = true;
                std::cout << "Writing sorted output to CSV file.\n";
            } else if (strcmp(argv[i], "-r") == 0) {
                repeat = std::stoi(argv[i + 1]);
                std::cout << "Repeat: " << repeat << "\n";
            } else if (strcmp(argv[i], "-n") == 0) {
                // Caps the data size by some integer n.
                max_n = std::stoi(argv[i + 1]);
                std::cout << "n: " << max_n << "\n";
            } else if (strcmp(argv[i], "-b") == 0) {
                benchmark_subfolder = argv[i + 1];
                std::cout << "Writing benchmarks to file: " << benchmark_subfolder << "\n";
            }
        }

        if (input == "") throw std::invalid_argument("Input digis input file missing");

        // Read CSV and load into raw array.
        auto vDigis = experimental::readCsv(input, repeat, max_n);
        std::cout << "CSV loaded." << "\n\n";

        // Copy digis to raw array.
        experimental::CbmStsDigiInput* aDigis = new experimental::CbmStsDigiInput[vDigis.size()];
        const size_t n = vDigis.size();
        std::copy(vDigis.begin(), vDigis.end(), aDigis);
        std::cout << "Copied array of size: " << n << "\n\n";

        // Benchmark.
        setenv("XPU_PROFILE", "1", 1); // always enable profiling in benchmark

        xpu::initialize();

        experimental::benchmark_runner runner(benchmark_subfolder);

        runner.add(new experimental::stdsort_bench(aDigis, n, writeOutput, checkResult));

        // Run block sort on all devices.
        runner.add(new experimental::blocksort_bench<experimental::BlockSort>(aDigis, n, writeOutput, checkResult));

        if (xpu::active_driver() != xpu::cpu) {
            std::cout << "Using GPU.\n\n";
            runner.add(new experimental::jansergeysort_bench<experimental::JanSergeySort>(aDigis, n, writeOutput, checkResult));
            runner.add(new experimental::jansergeysort_bench<experimental::JanSergeySortSimpleSum>(aDigis, n, writeOutput, checkResult, 1));
            //runner.add(new experimental::jansergeysort_bench<experimental::JanSergeySortParInsert>(aDigis, n, writeOutput, checkResult));
            // const CbmStsDigiInput* in_digis, const size_t in_n, const bool in_write = false, const bool in_check = true, unsigned int in_block_per_bucket = 2
            runner.add(new experimental::jansergeysort_bench<experimental::JanSergeySortSingleBlock>(aDigis, n, writeOutput, checkResult, 1));
        } else {
            // Only on
            std::cout << "No GPU device used.\n\n";
        }

        runner.run(10);

        delete[] aDigis;
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}
