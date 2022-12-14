#pragma once

#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <exception>
#include <iomanip> // put_time
#include <chrono>
#include <cstdlib>

#include "datastructures.h"

namespace experimental {

    void create_dir(const std::string dir) {
        const int dir_err = std::system(("mkdir -p " + dir).c_str());
        if (dir_err == -1) {
            std::cerr << "Error creating directory!n";
        }
    }

    bool file_exists(const std::string name) {
        std::ifstream f(name.c_str());
        return f.good();
    }

    bool file_empty(const std::string fileName) {
        std::ifstream infile(fileName);
        return infile.peek() == std::ifstream::traits_type::eof();
    }

    std::string get_env(std::string const& key) {
        char const* val = std::getenv(key.c_str()); 
        return val == NULL ? std::string() : std::string(val);
    }

    std::string get_filename(const std::string path) {
        std::string base_filename = path.substr(path.find_last_of("/") + 1);
        std::string::size_type const p(base_filename.find_last_of('.'));
        std::string file_without_extension = base_filename.substr(0, p);

        return file_without_extension;
    }

    std::string get_device() { return experimental::get_env("XPU_DEVICE"); }

    std::string stamp_name(const std::string filename, const std::string ext) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream datetime;
        datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
     
        return filename + "_" + datetime.str() + "." + ext;
    }

    std::vector <CbmStsDigiInput> readCsv(const std::string filename, const unsigned int repeat = 1, const unsigned int n = 0) {
        std::ifstream csv(filename);
        if (!csv.is_open()) {
            throw std::runtime_error("File: " + filename + " not found");
        }

        std::vector <CbmStsDigiInput> vDigis;

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
            // address,system,unit,ladder,half-ladder,module,sensor,side,channel,time,charge
            for (int i = 0; i < repeat; i++) {
                vDigis.push_back(CbmStsDigiInput(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9], cols[10]));

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
