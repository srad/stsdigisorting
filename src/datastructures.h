#pragma once

#include <string>
#include <unordered_map>
#include <iomanip>
#include "types.h"
#include "constants.h"

// Notice type alias last line.

namespace experimental {

    using address_t = int;

#ifdef DEBUG_SORT
    // This debug version is NOT suited for benchmarking, since the CbmStsDigi should be 8 bytes. 
    // The debug version carries the address, so the sorted result (in buckets) can be properly debugged.
    struct CbmStsDigi {
        int address;
        unsigned short channel;
        unsigned short charge;
        unsigned int time;

        CbmStsDigi(const int in_address, const unsigned short in_channel, const unsigned int in_time, const unsigned short in_charge) : address(in_address), channel(in_channel), time(in_time), charge(in_charge) {}
        CbmStsDigi() = default;
        ~CbmStsDigi() = default;

        std::string to_csv() { return std::to_string(address) + "," + std::to_string(channel) + "," + std::to_string(time) + "," + std::to_string(charge); }
        static std::string csv_headers() { return "address,channel,time"; }
    };
#else
    // Must be 8 byte for throughput bechmark.
    // Either: int channel, int time. Or: short channel, short charge, int time
    struct CbmStsDigi {
        unsigned short channel;
        unsigned short charge;
        unsigned int time;

        // Ignores address
        CbmStsDigi(const int in_address, const unsigned short in_channel, const unsigned int in_time, const unsigned short in_charge) : channel(in_channel), time(in_time), charge(in_charge) {}
        CbmStsDigi() = default;
        ~CbmStsDigi() = default;

        std::string to_csv() { return std::to_string(channel) + "," + std::to_string(time) + "," + std::to_string(charge); }
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
        int side;

        unsigned short channel;
        unsigned short charge;
        unsigned int time;

        CbmStsDigiInput() = default;

        CbmStsDigiInput(const int in_address, const int in_system, const int in_unit, const int in_ladder, const int in_half_ladder, const int in_module, const int in_sensor, const int in_side, const unsigned short in_channel, const unsigned int in_time, const unsigned short in_charge) : address(in_address), system(in_system), unit(in_unit), ladder(in_ladder), half_ladder(in_half_ladder), module(in_module), sensor(in_sensor), side(in_side), channel(in_channel), time(in_time), charge(in_charge) {}

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
        std::unordered_map<address_t, count_t> addressCounter;

        // Splits the input into multiple blocks, here 4.
        // +--------+------------+-----------+------------+
        // | 0..512 | 513..1024 | 1025..1536 | 1537..2047 |
        // +--------+------------+-----------+------------+
        std::array<unsigned short, 3> channelSegments = { 512, 1024, 1536 };

        // For each channel segment above, 
        std::unordered_map<address_t, std::array<count_t, 3>> channelSegmentCounter;
        
        /// <summary>
        /// Die Vector is NOT sorted, this is just the order that the digi addresses first appeared.
        /// Just used for the array layout to place the elements in a certain order. Which order is irrelevant.
        /// </summary>
        std::vector<address_t> addressOrder;

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

        // Contains for each bucket the index within the bucket, where the all channels >= 1024 starts.
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
        /// Running time: O(n) = 2*O(n) + 2*O(addressCount) = O(n) + O(1) = O(n)
        /// </summary>
        void createBuckets() {
            // -----------------------------------------------------------------------------------
            // 1. Init to zero and count all addresses. This will determine the output array layout.
            //    Each address bucket's size in the flat array is determined by each address count.
            // -----------------------------------------------------------------------------------
            for (int i = 0; i < n_; i++) {                
                // Init map entries.
                if (addressCounter.find(input[i].address) == addressCounter.end()) {
                    addressCounter[input[i].address] = 0;
                    addressStartIndex[input[i].address] = 0;

                    // Just take the addresses just in the order they first appear to use them as buckets.
                    addressOrder.push_back(input[i].address);

                    channelSegmentCounter[input[i].address] = { 0 };
                }

                // Count the occurence of each address.
                addressCounter[input[i].address]++;
                
                // Could also be a loop with unrolling.
                // Count the channel count for each interval: 0, 512, 1024, 1536, 2047
                // +--------+-----------+------------+------------+
                // | 0..512 | 513..1024 | 1025..1536 | 1537..2047 |
                // +--------+-----------+------------+------------+
                //         i0          i1            i2
                //
                // channelSegmentCounter[0] = i0
                // channelSegmentCounter[1] = i1
                // channelSegmentCounter[2] = i2
                // 
                channelSegmentCounter[input[i].address][0] += (input[i].channel >= 0                        && input[i].channel <= channelSegments[1]) * 1; //        0, 512
                channelSegmentCounter[input[i].address][1] += (input[i].channel >= (channelSegments[1] + 1) && input[i].channel <= channelSegments[2]) * 1; //  512 + 1, 1024
                channelSegmentCounter[input[i].address][2] += (input[i].channel >= (channelSegments[2] + 1) && input[i].channel <= channelSegments[3]) * 1; // 1024 + 1, 1536
            }
            
            // -----------------------------------------------------------------------------------
            // 2. Excluside sum per address. O(addressOrder.size()) -> small.
            //
            //    This defines the output layout. Each address bucket's size is defined by the
            //    count the address occured.
            // -----------------------------------------------------------------------------------
            int sum = 0;
            // The start index of the first address is = 0.
            for (int i = 0; i < addressOrder.size(); i++) {
                const address_t address = addressOrder[i];

                addressStartIndex[address] = sum;
                sum += addressCounter[address];
                // std::cout << address << ":" << addressStartIndex[address] << "\n";
            }

            // -----------------------------------------------------------------------------------
            // 3. Compute start and end indexes: O(addressOrder.size()) -> small.
            // -----------------------------------------------------------------------------------
            addresses_ = new address_t[addressOrder.size()];

            startIndex = new index_t[addressOrder.size()];
            endIndex = new index_t[addressOrder.size()];

            // Three segments for each address bucket.
            channelSplitIndex = new index_t[addressOrder.size() * 3];

            bucketCount_ = addressOrder.size();

            for (int i = 0; i < addressOrder.size(); i++) {
                addresses_[i] = addressOrder[i];

                // The bucket's start and end indexes. This are NOT the segment indexes.
                startIndex[i] = addressStartIndex[addressOrder[i]];
                endIndex[i] = startIndex[i] + addressCounter[addressOrder[i]] - 1;

                // Each bucket naturally starts at index 0 (+offset) and is divided into segments.
                //                                                                                  // Start of first segment implicitly start from channels [0..512]
                channelSplitIndex[i]     = startIndex[i] + channelSegmentCounter[addresses_[i]][0]; // Start of second segments, with all digis with channels [513..1024]
                channelSplitIndex[i + 1] = startIndex[i] + channelSegmentCounter[addresses_[i]][1]; // Start of third segments,  with all digis with channels [1025..1536]
                channelSplitIndex[i + 2] = startIndex[i] + channelSegmentCounter[addresses_[i]][2]; // Start of third segments,  with all digis with channels [1537..2047]
            }

            // -----------------------------------------------------------------------------------
            // 4. Place in virtual buckets in the flat array.
            // Copy elements to the right location
            // -----------------------------------------------------------------------------------
            for (int i = 0; i < n_; i++) {
                // Compute the index: 0, 1, 2 based on the channel interval.
                // Could also if the hell out of this, but make it branch free.
                // Segment channels: 512, 1024, 1536
                const auto segmentIndex = (input[i].channel > channelSegments[0]) + (input[i].channel > channelSegments[1]) + (input[i].channel > channelSegments[2]);

                // All digis on the back-side (>= 1024) are offset by the number of front-side digis in the bucket (<= 1023).
                // If the DEBUG_SORT symbol is not defined, the address in the CbmStsDigi constructor is ignored and not part of the type.
                digis[addressStartIndex[input[i].address] + (channelSegmentCounter[input[i].address][segmentIndex]++)] = CbmStsDigi(input[i].address, input[i].channel, input[i].time, input[i].charge);
            }
        }
    };
}

using digi_t = experimental::CbmStsDigi;
using bucket_t = experimental::CbmStsDigiBucket;