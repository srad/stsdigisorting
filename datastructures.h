#pragma once

#include <string>
#include <unordered_map>
#include <iomanip>

namespace experimental {

    class CbmStsDigi {
    public:
        // address,system,unit,ladder,half-ladder,module,sensor,side,channel,time
        int address;
        int system;
        int unit;
        int ladder;
        int half_ladder;
        int module;
        int sensor;
        int side; // aufteilung in doppelte blockzahl und vorder/rückseite getrennt bearbeiten
        // alle channels >= 1024 sind die rückseite. -> blocks verdoppeln
        int channel;
        int time;

        CbmStsDigi() = default;
        CbmStsDigi(int in_address, int in_system, int in_unit, int in_ladder, int in_half_ladder, int in_module, int in_sensor, int in_side, int in_channel, int in_time) : address(in_address), system(in_system), unit(in_unit), ladder(in_ladder), half_ladder(in_half_ladder), module(in_module), sensor(in_sensor), side(in_side), channel(in_channel), time(in_time) {}

        std::string to_string() { return "(address: " + to_zero_lead(address, 10) + ", channel: " + to_zero_lead(channel, 4) + ", time: " + to_zero_lead(time, 5) + ")"; }

    private:
        static std::string to_zero_lead(const int value, const unsigned precision) {
            std::ostringstream oss;
            oss << std::setw(precision) << std::setfill('0') << value;
            return oss.str();
        }
    };

    /// <summary>
    /// The purpose of this class is to have a flat array that contains virtual buckets
    /// specified by start and end indexes for each addresses. The point is to copy the data structure
    /// -as is- to the GPU for further computation.
    /// </summary>
    class CbmStsDigiBucket {
        int* addresses_;
        std::unordered_map<int, int> addressCounter; // O(1)

        /// <summary>
        /// Die Vector is NOT sorted, this is just the order that the digi addresses first appeared.
        /// Just used for the array layout to place the elements in a certain order. Which order is irrelevant.
        /// </summary>
        std::vector<int> addressOrder;

        std::unordered_map<int, int> prefixSum;
        CbmStsDigi* input;
        size_t n_;
        int bucketCount_;

    public:
        // Contains after construction the bucket with digis.
        CbmStsDigi* digis;

        // Start and end indexes (not size) of digis.
        int* startIndex;
        int* endIndex;

        CbmStsDigiBucket(const CbmStsDigi* in_digis, const size_t in_n) : n_(in_n), digis(new CbmStsDigi[in_n]), input(new CbmStsDigi[in_n]) {
            std::copy(in_digis, in_digis + in_n, input);
            createBuckets();
        }
        
        ~CbmStsDigiBucket() {
            delete[] digis;
            delete[] input;
            delete[] startIndex;
            delete[] endIndex;
            delete[] addresses_;
        }

        CbmStsDigi& operator[](int i) { return digis[i]; }

        int size() const { return bucketCount_; }
        size_t n() const { return n_; }

        int* address() const { return addresses_;  }
        int getAddress(const int i) const { return addresses_[i]; }

        int begin(const int i) const { return startIndex[i]; }
        int end(const int i) const { return endIndex[i]; }

        std::string to_index_string(int i) { return "(address: " + std::to_string(addresses_[i]) + ", start-idx: " + std::to_string(startIndex[i]) + ", end-idx:" + std::to_string(endIndex[i]) + ")"; }

    private:
        /// <summary>
        /// Takes O(n) = 2*O(n) + 2*O(addressCount)
        /// </summary>
        void createBuckets() {
            // 1. Init to zero and count addresses.
            for (int i = 0; i < n_; i++) {
                // Init map entries.
                if (addressCounter.find(input[i].address) == addressCounter.end()) {
                    addressCounter[input[i].address] = 0;
                    prefixSum[input[i].address] = 0;
                    // Just for simpler and quicker traversal.
                    addressOrder.push_back(input[i].address);
                }
                addressCounter[input[i].address] += 1;
            }

            // 2. Prefix sum per address. O(addressOrder.size()) -> small.
            int sum = 0;
            // The address of the first element is = 0.
            for (int i = 0; i < addressOrder.size(); i++) {
                prefixSum[addressOrder[i]] = sum;
                sum += addressCounter[addressOrder[i]];
                //std::cout << ", address: " << address << ", prefixSum: " << prefixSum[addressOrder[i]] << ", count: " << addressCounter[addressOrder[i]] << "\n";
            }

            // 3. Compute start and end indexes: O(addressOrder.size()) -> small.
            addresses_ = new int[addressOrder.size()];
            startIndex = new int[addressOrder.size()];
            endIndex = new int[addressOrder.size()];
            bucketCount_ = addressOrder.size();

            for (int i = 0; i < addressOrder.size(); i++) {
                addresses_[i] = addressOrder[i];
                startIndex[i] = prefixSum[addressOrder[i]];
                endIndex[i] = prefixSum[addressOrder[i]] + addressCounter[addressOrder[i]] - 1;
            }

            // 4. Place in virtual buckets in the flat array.
            // Copy elements to the right location
            for (int i = 0; i < n_; i++) {
                digis[prefixSum[input[i].address]++] = input[i];
            }
        }
    };
}
