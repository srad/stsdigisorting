#include <iostream>
#include <cstring>
#include "common.h"

#include "benchmark/blocksort.h"
#include "benchmark/stdsort.h"
#include "benchmark/jansergeysort.h"

#include "sorting/SortKernel.h"
#include "sorting/JanSergeySort.h"

int main(int argc, char** argv) {
    experimental::CbmStsDigi* aDigis;

    try {
        // Command line paraams.
        std::string input;
        std::string output;
        int repeat = 1;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0) {
                input = argv[i + 1];
                std::cout << "Input: " << input << "\n";
            } else if (strcmp(argv[i], "-o") == 0) {
                output = argv[i + 1];
                std::cout << "Output: " << output << "\n";
            } else if (strcmp(argv[i], "-r") == 0) {
                repeat = std::stoi(argv[i + 1]);
                std::cout << "Repeat: " << repeat << "\n";
            }
        }

        if (input == "") throw std::invalid_argument("Input digis input file missing");

        // Read CSV and load into raw array.
        auto vDigis = experimental::readCsv(input, repeat);
        std::cout << "CSV loaded." << "\n";

        aDigis = vDigis.data();
        const size_t n = vDigis.size();
        vDigis.clear();

        // Benchmark.
        setenv("XPU_PROFILE", "1", 1); // always enable profiling in benchmark

        xpu::initialize();

        benchmark_runner runner;

//        if (xpu::active_driver() != xpu::cpu) {
        runner.add(new stdsort_bench(aDigis, n));
        runner.add(new blocksort_bench<BlockSort>(aDigis, n));
        runner.add(new jansergeysort_bench<JanSergeySort>(aDigis, n));
        //      }

        runner.run(10);
    }
    catch (std::exception& e) {
        delete[] aDigis;
        std::cerr << e.what() << "\n";
        return 1;
    }

    delete[] aDigis;

    return 0;
}
