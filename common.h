#pragma once

#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <exception>

#include "datastructures.h"

namespace experimental {

    constexpr int channelCount = 2048;

    std::vector <CbmStsDigi> readCsv(const std::string filename, const unsigned int repeat = 1) {
        std::ifstream csv(filename);
        if (!csv.is_open()) {
            throw std::runtime_error("File: " + filename + " not found");
        }

        std::vector <CbmStsDigi> vDigis;

        std::string line;

        for (int i = 0; i < repeat; i++) {
            // Skip header
            std::getline(csv, line);
            while (std::getline(csv, line)) {
                std::stringstream ss(line);
                std::vector<int> cols;

                while (ss.good()) {
                    std::string substr;
                    std:
                    getline(ss, substr, ',');
                    cols.push_back(std::stoi(substr));
                }

                // address,system,unit,ladder,half-ladder,module,sensor,side,channel,time
                vDigis.push_back(CbmStsDigi(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9]));
            }
            csv.clear();
            csv.seekg(0);
        }
        csv.close();

        return vDigis;
    }
}