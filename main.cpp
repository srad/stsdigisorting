#include <iostream>
#include <cstring>
#include "common.h"

#include "benchmark/blocksort.h"
#include "benchmark/stdsort.h"

#include "sorting/SortKernel.h"

int main(int argc, char** argv) {
    try {
        std::string input;
        std::string output;
        int repeat = 1;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i],"-i")==0) {
                input = argv[i+1];
                std::cout << "Input: " << input << "\n";
            } else if (strcmp(argv[i],"-o") == 0) {
                output = argv[i+1];
                std::cout << "Output: " << output << "\n";
            } else if (strcmp(argv[i],"-r") == 0) {
                repeat = std::stoi(argv[i+1]);
                std::cout << "Repeat: " << repeat << "\n";
            }
        }

        if (input == "") throw std::invalid_argument("Input digis input file missing");

        auto vDigis = experimental::readCsv(input, repeat);
        std::cout << "CSV loaded." << "\n";

        experimental::CbmStsDigi* aDigis = vDigis.data();
        auto n = vDigis.size();
        vDigis.clear();

        experimental::CbmStsDigiBucket buckets(aDigis, n);
        std::cout << "Buckets created." << "\n";

	// Benchmark
        setenv("XPU_PROFILE", "1", 1); // always enable profiling in benchmark

        xpu::initialize();

        benchmark_runner runner;

//        if (xpu::active_driver() != xpu::cpu) {
            runner.add(new stdsort_bench(aDigis, n));
            runner.add(new blocksort_bench<BlockSort>(aDigis, n));

  //      }

        runner.run(10);
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}
