#include <iostream>
#include <cstring>
#include <cstdlib>
#include "common.h"

#include "benchmark/blocksort.h"
#include "benchmark/stdsort.h"
#include "benchmark/jansergeysort.h"
//#include "benchmark/jansergeysort_hip.h"

#include "sorting/SortKernel.h"
#include "sorting/JanSergeySort.h"

int main(int argc, char** argv) {
    try {
        // Command line params.
        std::string input;
        std::string output;
        unsigned int repeat = 1;
        unsigned int max_n = 0;
        bool writeOutput = false;
        bool checkResult = false;

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
            }
        }

        if (input == "") throw std::invalid_argument("Input digis input file missing");

        // Read CSV and load into raw array.
        auto vDigis = experimental::readCsv(input, repeat, max_n);
        std::cout << "CSV loaded." << "\n\n";

        // Copy digis to raw array.
        experimental::CbmStsDigi* aDigis = new experimental::CbmStsDigi[vDigis.size()];
        const size_t n = vDigis.size();
        std::copy(vDigis.begin(), vDigis.end(), aDigis);
        std::cout << "Copied array of size: " << n << "\n\n";

        // Benchmark.
        setenv("XPU_PROFILE", "1", 1); // always enable profiling in benchmark

        xpu::initialize();

        benchmark_runner runner;

        // Run block sort on all devices.
        runner.add(new blocksort_bench<BlockSort>(aDigis, n, writeOutput, checkResult));

        if (xpu::active_driver() != xpu::cpu) {
            std::cout << "Using GPU.\n\n";
            runner.add(new jansergeysort_bench<JanSergeySort>(aDigis, n, writeOutput, checkResult));
        } else {
            // Only on
            runner.add(new stdsort_bench(aDigis, n, writeOutput, checkResult));
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
