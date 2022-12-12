#pragma once

#include <xpu/device.h>
#include "constants.h"

namespace experimental {

    /*

    // -------------------------------------------------------------------
    // CbmStsAddress.cxx
    // -------------------------------------------------------------------

    enum EStsElementLevel {
        kStsSystem,
        kStsUnit,
        kStsLadder,
        kStsHalfLadder,
        kStsModule,
        kStsSensor,
        kStsSide,
        kStsNofLevels
    };

    const int32_t kVersionSize  = 4;   // Bits for version number
    const int32_t kVersionShift = 28;  // First bit for version number
    const int32_t kVersionMask  = (1 << kVersionSize) - 1;

    const uint32_t kCurrentVersion = 1;

  // clang-format off
  // -----    Definition of address bit field   ------------------------------
  const uint16_t kBits[kCurrentVersion + 1][kStsNofLevels] = {
    // Version 0 (until 23 August 2017)
    {
      4,  // system
      4,  // unit / station
      4,  // ladder
      1,  // half-ladder
      3,  // module
      2,  // sensor
      1   // side
    },

    // Version 1 (current, since 23 August 2017)
    {
      4,  // system
      6,  // unit
      5,  // ladder
      1,  // half-ladder
      5,  // module
      4,  // sensor
      1   // side
    }

  };
  // -------------------------------------------------------------------------

    // -----    Bit shifts -----------------------------------------------------
    const int32_t kShift[kCurrentVersion + 1][kStsNofLevels] = {
        {0, kShift[0][0] + kBits[0][0], kShift[0][1] + kBits[0][1], kShift[0][2] + kBits[0][2], kShift[0][3] + kBits[0][3],
        kShift[0][4] + kBits[0][4], kShift[0][5] + kBits[0][5]},

        {0, kShift[1][0] + kBits[1][0], kShift[1][1] + kBits[1][1], kShift[1][2] + kBits[1][2], kShift[1][3] + kBits[1][3],
        kShift[1][4] + kBits[1][4], kShift[1][5] + kBits[1][5]}};
    // -------------------------------------------------------------------------


    // -----    Bit masks  -----------------------------------------------------
    const int32_t kMask[kCurrentVersion + 1][kStsNofLevels] = {
        {(1 << kBits[0][0]) - 1, (1 << kBits[0][1]) - 1, (1 << kBits[0][2]) - 1, (1 << kBits[0][3]) - 1,
        (1 << kBits[0][4]) - 1, (1 << kBits[0][5]) - 1, (1 << kBits[0][6]) - 1},

        {(1 << kBits[1][0]) - 1, (1 << kBits[1][1]) - 1, (1 << kBits[1][2]) - 1, (1 << kBits[1][3]) - 1,
        (1 << kBits[1][4]) - 1, (1 << kBits[1][5]) - 1, (1 << kBits[1][6]) - 1}};
    // -------------------------------------------------------------------------

    uint32_t GetVersion(const int32_t address) {
        return uint32_t((address & (kVersionMask << kVersionShift)) >> kVersionShift);
    }

    uint32_t GetElementId(const int32_t address, const EStsElementLevel level) {
        uint32_t version = GetVersion(address);
        return (address & (kMask[version][level] << kShift[version][level])) >> kShift[version][level];
    }

    //   ss << "StsAddress: address " << address << " (version " << GetVersion(address) << ")"
    //      << ": system " << GetElementId(address, kStsSystem) << ", unit " << GetElementId(address, kStsUnit) << ", ladder "
    //      << GetElementId(address, kStsLadder) << ", half-ladder " << GetElementId(address, kStsHalfLadder) << ", module "
    //      << GetElementId(address, kStsModule) << ", sensor " << GetElementId(address, kStsSensor) << ", side "
    //      << GetElementId(address, kStsSide);

    // -------------------------------------------------------------------
    // end CBM source
    // -------------------------------------------------------------------

    XPU_D void group_by_address(CbmStsDigiInput* digis, const unsigned int n) {
        // 64+32+2+32+16+2=148
        constexpr unsigned int addressSpace = 64 + 32 + 2 + 32 + 16 + 2;
        unsigned int map[addressSpace];

        for (int i = xpu::thread_idx::x(); i < n; i += xpu::block_dim::x()) {
            //smem.channelOffset[i] = 0;
            const auto unit = GetElementId(digis[i].address, EStsElementLevel::kStsUnit);
            const auto ladder = GetElementId(digis[i].address, EStsElementLevel::kStsLadder);
            const auto hLadder = GetElementId(digis[i].address, EStsElementLevel::kStsHalfLadder);
            const auto module = GetElementId(digis[i].address, EStsElementLevel::kStsModule);
            const auto sensor = GetElementId(digis[i].address, EStsElementLevel::kStsSensor);
            const auto side = GetElementId(digis[i].address, EStsElementLevel::kStsSide);
        }
        xpu::barrier();
    } 
    */

    template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    XPU_D void prescan(T* data, T* temp) {
        const int n = experimental::channelCount;
        const int countOffset = 0;

        int thid = xpu::thread_idx::x();
        int offset = 1;

        // load input into shared memory
        temp[2 * thid] = data[countOffset + 2 * thid];
        temp[2 * thid + 1] = data[countOffset + 2 * thid + 1];

        // build sum in place up the tree
        for (int d = n >> 1; d > 0; d >>= 1) {
            xpu::barrier();
            if (thid < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                temp[bi] += temp[ai];
            }
            offset *= 2;
        }

        if (thid == 0) { temp[n - 1] = 0; } // clear the last element

        // traverse down tree & build scan
        for (int d = 1; d < n; d *= 2) {
            offset >>= 1;

            xpu::barrier();

            if (thid < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }

        xpu::barrier();

        data[countOffset + 2 * thid] = temp[2 * thid]; // write results to device memory
        data[countOffset + 2 * thid + 1] = temp[2 * thid + 1];
    }

    constexpr int sideSeperator = 1024;

    XPU_D int binary_search(experimental::CbmStsDigi* list, int length, int to_be_found){
        int p = 0;
        int r = length - 1;
        int q = (r + p) / 2;
        int counter = 0;

        while (p <= r) {
            counter++;
            if (list[q].channel == to_be_found) {
                return q;
            }
            else {
                if (list[q].channel <= to_be_found) {
                    p = q + 1;
                    q = (r + p) / 2;
                }
                else {
                    r = q - 1;
                    q = (r + p) / 2;    
                }
            }
        }
        return -1;
    }

    // See: https://stackoverflow.com/questions/6553970/find-the-first-element-in-a-sorted-array-that-is-greater-than-the-target
    XPU_D int findSideSeperatorIndex(const experimental::CbmStsDigi* arr, const int n, const int target) {
        int low = 0;
        int high = n;

        while (low != high) {
            int mid = (low + high) / 2;

            if (arr[mid].channel <= target) {
                /* This index, and everything below it, must not be the first element
                * greater than what we're looking for because this element is no greater
                * than the element.
                */
                low = mid + 1;
            }
            else {
                /* This element is at least as large as the element, so anything after it can't
                * be the first element that's at least as large.
                */
                high = mid;
            }
        }

        /* Now, low and high both point to the element in question. */
        return high;
    }

} // namespace