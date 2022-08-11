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

    std::vector <CbmStsDigi> readCsv(const std::string filename, const unsigned int repeat = 1, const unsigned int n = 0) {
        std::ifstream csv(filename);
        if (!csv.is_open()) {
            throw std::runtime_error("File: " + filename + " not found");
        }

        std::vector <CbmStsDigi> vDigis;

        std::string line;

        // Skip header
        std::getline(csv, line);

        unsigned int cnt = 0;
        while (std::getline(csv, line)) {
            std::stringstream ss(line);
            std::vector<int> cols;

            while (ss.good()) {
                std::string substr;
                std::getline(ss, substr, ',');
                cols.push_back(std::stoi(substr));
            }

            // Artificial duplication of data for testing purposes.
            // address,system,unit,ladder,half-ladder,module,sensor,side,channel,time
            for (int i = 0; i < repeat; i++) {
                vDigis.push_back(CbmStsDigi(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9]));

                cnt++;
                if (n != 0 && cnt == n) {
                    goto out;
                }
            }
        }
out:
        csv.close();

        return vDigis;
    }
}
