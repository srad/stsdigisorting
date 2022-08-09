#include <iostream>
#include <cstring>
#include "common.h"

int main(int argc, char** argv) {
    try {
        std::string input;
        std::string output;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i],"-i")==0) {
                input = argv[i+1];
                std::cout << "Input: " << input << "\n";
            } else if (strcmp(argv[i],"-o") == 0) {
                output = argv[i+1];
                std::cout << "Output: " << output << "\n";
            }
        }

        if (input == "") throw std::invalid_argument("Input digis input file missing");

        auto vDigis = experimental::readCsv(input, 1);
        std::cout << "CSV loaded." << "\n";
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
    }
}