#include "../datastructures.h"

// Include host functions to control the GPU.
#include <xpu/host.h>
#include "../benchmark/benchmark.h"

#include <iostream>
#include <random>
#include <vector>

template<typename Kernel>
class jansergeysort_bench : public benchmark {

    const size_t n;

    // Big difference here is that the digis are grouped in buckets and then bucket-wise sorted.
    experimental::CbmStsDigiBucket bucket;

    // Write the sorted result to CSV file.
    std::ofstream output;

    const experimental::CbmStsDigi* digis;
    xpu::hd_buffer<experimental::CbmStsDigi> buffDigis;
    xpu::hd_buffer<experimental::CbmStsDigi> buffOutput;

    xpu::hd_buffer<int> buffStartIndex;
    xpu::hd_buffer<int> buffEndIndex;

public:
    jansergeysort_bench(const experimental::CbmStsDigi* in_digis, const size_t in_n) : n(in_n), digis(new experimental::CbmStsDigi[in_n]) {
        std::copy(in_digis, in_digis + in_n, digis);
    }

   ~jansergeysort_bench() { delete[] digis; }

    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        buffDigis = xpu::hd_buffer<experimental::CbmStsDigi>(n);
        buffOutput = xpu::hd_buffer<experimental::CbmStsDigi>(n);

        output.open("jansergey_sort_output.csv", std::ios::out | std::ios::trunc);

        bucket = experimental::CbmStsDigiBucket(digis, n);
        std::cout << "Buckets created." << "\n";

        buffStartIndex = xpu::hd_buffer<experimental::CbmStsDigi>(bucket.size());
        buffEndIndex = xpu::hd_buffer<experimental::CbmStsDigi>(bucket.size());


        std::copy(bucket.startIndex, bucket.startIndex + bucket.size(), buffStartIndex.h());
        std::copy(bucket.endIndex, bucket.endIndex + bucket.size(), buffEndIndex.h());

        std::copy(bucket.digis, bucket.digis + n, buffDigis.h());
    }

    void teardown() {
        check();

	output << "index,address,channel,time\n";

        for (int i=0; i < n; i++) {
            output << i << "," << buffOutput.h()[i].address << "," << buffOutput.h()[i].channel << "," << buffOutput.h()[i].time << "\n";
        }

        output.close();
        buffDigis.reset();
        buffOutput.reset();
    }

    void run() {
        xpu::copy(buffDigis, xpu::host_to_device);
        // buckets.digis, buckets.size(), n, buckets.startIndex, buckets.endIndex
        xpu::run_kernel<Kernel>(n, buffDigis.d(), bucket.size(), buffStartIndex.d(), buffEndIndex.d(), buffOutput.d());
        xpu::copy(buffOutput, xpu::device_to_host);
    }

    void check() {
        // Check if data is sorted.
        bool ok = true;

        // Start at second element and compare to previous for all i.
        for (int i=1; i < n; i++) {
            bool okThisRun = true;

            const auto& curr = buffOutput.h()[i];
            const auto& prev = buffOutput.h()[i - 1];

            // Digi i has always a >= bigger channel number after sorting than digi i-1.
            ok &= curr.channel >= prev.channel;
            okThisRun &= curr.channel >= prev.channel;

            // Only check the time_j < time_j-1 within the same channels.
            if (curr.channel == prev.channel) {
                auto faa = curr.time >= prev.time;
                ok &= faa;
                okThisRun &= curr.time >= prev.time;
            }

            if (!okThisRun) {
                std::cout << "Error: " << "\n";
                printf("(%d/%d): (%d, %d, %d)\n", i, n, prev.address, prev.channel, prev.time);
                printf("(%d/%d): (%d, %d, %d)\n", i, n, curr.address, curr.channel, curr.time);
            }
        }

        if (ok) {
            std::cout << "Data is sorted!" << std::endl;
        } else {
            std::cout << "Error: Data is not sorted!" << std::endl;
        }
    }

    size_t bytes() { return n * sizeof(experimental::CbmStsDigi); }
    std::vector<float> timings() { return xpu::get_timing<Kernel>(); }

};
