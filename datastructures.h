#pragma once

#include <string>
#include <unordered_map>
#include <iomanip>
#include "types.h"
#include "constants.h"

// Notice type alias last line.

namespace experimental {

    using address_t = unsigned int;
    using address_counter_t = std::unordered_map<address_t, count_t>;

    // For the index calculation we need to know for each address bucket, how many
    // digis are of channel < 1024. Then the layout is known:
    // [bucket_start_index .. front_end_index back_start_index .. bucket_end_index]
    // front_end_index = number of digis in bucket with channel < 1024.
    using channel_side_counter_t = std::unordered_map<address_t, std::array<count_t, 2>>;


    // The debug version carries the address, so the sorting result (in buckets) can be better debugged.
#ifdef DEBUG_SORT
    struct CbmStsDigi {
        int address;
        int channel;
        int time;

        CbmStsDigi(int in_address, int in_channel, int in_time) : address(in_address), channel(in_channel), time(in_time) {}
        CbmStsDigi() = default;
        ~CbmStsDigi() = default;

        std::string to_csv() { return std::to_string(address) + "," + std::to_string(channel) + "," + std::to_string(time); }
        static std::string csv_headers() { return "address,channel,time"; }
    };
#else
    struct CbmStsDigi {
        int channel;
        int time;

        CbmStsDigi(int in_address, int in_channel, int in_time) : channel(in_channel), time(in_time) {}
        CbmStsDigi() = default;
        ~CbmStsDigi() = default;

        std::string to_csv() { return std::to_string(channel) + "," + std::to_string(time); }
        static std::string csv_headers() { return "channel,time"; }
    };
#endif

    // This is the data that is read from the CSV file. It contains more information than is relevant for the sorting benchmark.
    struct CbmStsDigiInput {
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

        CbmStsDigiInput() = default;

        CbmStsDigiInput(int in_address, int in_system, int in_unit, int in_ladder, int in_half_ladder, int in_module, int in_sensor, int in_side, int in_channel, int in_time) : address(in_address), system(in_system), unit(in_unit), ladder(in_ladder), half_ladder(in_half_ladder), module(in_module), sensor(in_sensor), side(in_side), channel(in_channel), time(in_time) {}

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
        address_t* addresses_;
        address_counter_t addressCounter;
        channel_side_counter_t channelSideCounter;

        const unsigned int front = 0;
        const unsigned int back = 1;

        // Doppelte bucket anzahl, vor/rückseite

        /// <summary>
        /// Die Vector is NOT sorted, this is just the order that the digi addresses first appeared.
        /// Just used for the array layout to place the elements in a certain order. Which order is irrelevant.
        /// </summary>
        std::vector<count_t> addressOrder;

        std::unordered_map<address_t, count_t> addressStartIndex;
        CbmStsDigiInput* input;
        size_t n_;
        count_t bucketCount_;

    public:
        // Contains after construction the bucket with digis.
        CbmStsDigi* digis;

        // Start and end indexes (not size) of digis.
        index_t* startIndex;
        index_t* endIndex;

        index_t* channelSplitIndex;

        CbmStsDigiBucket(const CbmStsDigiInput* in_digis, const size_t in_n) : n_(in_n), digis(new CbmStsDigi[in_n]), input(new CbmStsDigiInput[in_n]) {
            std::copy(in_digis, in_digis + in_n, input);
            createBuckets();
        }

        ~CbmStsDigiBucket() {
            delete[] digis;
            delete[] input;
            delete[] startIndex;
            delete[] endIndex;
            delete[] addresses_;
            delete[] channelSplitIndex;
        }

        CbmStsDigi& operator[](int i) { return digis[i]; }

        count_t size() const { return bucketCount_; }

        size_t n() const { return n_; }

        address_t* address() const { return addresses_; }

        address_t getAddress(const int i) const { return addresses_[i]; }

        index_t begin(const int i) const { return startIndex[i]; }

        index_t end(const int i) const { return endIndex[i]; }

        index_t frontBegin(const int i) const { return begin(i); }
        index_t frontEnd(const int i) const { return channelSplitIndex[i]; }

        index_t backBegin(const int i) const { return channelSplitIndex[i] + 1; }
        index_t backEnd(const int i) const { return end(i); }

        std::string to_index_string(index_t i) { return "(address: " + std::to_string(addresses_[i]) + ", start-idx: " + std::to_string(startIndex[i]) + ", end-idx:" + std::to_string(endIndex[i]) + ")"; }

    private:
        /// <summary>
        /// Takes O(n) = 2*O(n) + 2*O(addressCount)
        /// </summary>
        void createBuckets() {

            // -----------------------------------------------------------------------------------
            // 1. Init to zero and count addresses.
            // -----------------------------------------------------------------------------------
            for (int i = 0; i < n_; i++) {                
                // Init map entries.
                if (addressCounter.find(input[i].address) == addressCounter.end()) {
                    addressCounter[input[i].address] = 0;
                    addressStartIndex[input[i].address] = 0;

                    // Just take the addresses just in the order they first appear to use them as buckets.
                    addressOrder.push_back(input[i].address);

                    channelSideCounter[input[i].address] = { 0 };
                }

                addressCounter[input[i].address]++;
                
                // The front index starts always at addressStartIndex + 0++,
                // only the back is offset by the number of indexes at the front.
                channelSideCounter[input[i].address][back] += (input[i].channel < 1024) * 1;
            }
            
            // -----------------------------------------------------------------------------------
            // 2. Excluside sum sum per address. O(addressOrder.size()) -> small.
            // -----------------------------------------------------------------------------------
            int sum = 0;
            // The address of the first element is = 0.
            for (int i = 0; i < addressOrder.size(); i++) {
                const address_t address = addressOrder[i];

                addressStartIndex[address] = sum;
                sum += addressCounter[address];
            }

            // -----------------------------------------------------------------------------------
            // 3. Compute start and end indexes: O(addressOrder.size()) -> small.
            // -----------------------------------------------------------------------------------
            addresses_ = new index_t[addressOrder.size()];

            startIndex = new index_t[addressOrder.size()];
            endIndex = new index_t[addressOrder.size()];

            // At which index start all digis with channel > 1024.
            channelSplitIndex = new index_t[addressOrder.size()];

            bucketCount_ = addressOrder.size();

            for (int i = 0; i < addressOrder.size(); i++) {
                addresses_[i] = addressOrder[i];
                startIndex[i] = addressStartIndex[addressOrder[i]];
                endIndex[i] = startIndex[i] + addressCounter[addressOrder[i]] - 1;
                channelSplitIndex[i] = startIndex[i] + channelSideCounter[addresses_[i]][back];
            }

            // -----------------------------------------------------------------------------------
            // 4. Place in virtual buckets in the flat array.
            // Copy elements to the right location
            // -----------------------------------------------------------------------------------
            for (int i = 0; i < n_; i++) {
                digis[addressStartIndex[input[i].address] + (channelSideCounter[input[i].address][(input[i].channel >= 1024)]++)] = CbmStsDigi(input[i].address, input[i].channel, input[i].time);
            }
        }
    };
}

using digi_t = experimental::CbmStsDigi;
using bucket_t = experimental::CbmStsDigiBucket;