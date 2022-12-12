#include <cstdint>
#include <cassert>

#include "AddressBuckets.h"
#include "../datastructures.h"
#include "../common.h"
#include "../device.h"
#include "../types.h"

XPU_IMAGE(experimental::AddressBucketsKernel);

namespace experimental {

    // Converted from enum to this.
    // Must correspond to the entries below.
    struct EStsElementLevel
    {
        static constexpr int32_t kStsSystem = 0;
        static constexpr int32_t kStsUnit = 1;
        static constexpr int32_t kStsLadder = 2;
        static constexpr int32_t kStsHalfLadder = 3;
        static constexpr int32_t kStsModule = 4;
        static constexpr int32_t kStsSensor = 5;
        static constexpr int32_t kStsSide = 6;
        static constexpr int32_t kStsNofLevels = 7;
    };

    constexpr int32_t kVersionSize = 4;   // Bits for version number
    constexpr int32_t kVersionShift = 28;  // First bit for version number
    constexpr int32_t kVersionMask = (1 << kVersionSize) - 1;
    constexpr uint32_t kCurrentVersion = 1;

    // clang-format off
    // -----    Definition of address bit field   ------------------------------
    constexpr uint16_t kBits[kCurrentVersion + 1][EStsElementLevel::kStsNofLevels] = {
        // Version 0 (until 23 August 2017)
        {
         4,				// system
         4,				// unit / station
         4,				// ladder
         1,				// half-ladder
         3,				// module
         2,				// sensor
         1				// side
         },

        // Version 1 (current, since 23 August 2017)
        {
         4,				// system
         6,				// unit
         5,				// ladder
         1,				// half-ladder
         5,				// module
         4,				// sensor
         1				// side
         }
    };

    // -----    Bit shifts -----------------------------------------------------
    constexpr int32_t kShift[kCurrentVersion + 1][EStsElementLevel::kStsNofLevels] = {
      {0, kShift[0][0] + kBits[0][0], kShift[0][1] + kBits[0][1],
       kShift[0][2] + kBits[0][2], kShift[0][3] + kBits[0][3],
       kShift[0][4] + kBits[0][4], kShift[0][5] + kBits[0][5]},

      {0, kShift[1][0] + kBits[1][0], kShift[1][1] + kBits[1][1],
       kShift[1][2] + kBits[1][2], kShift[1][3] + kBits[1][3],
       kShift[1][4] + kBits[1][4], kShift[1][5] + kBits[1][5]}
    };

    // -----    Bit masks  -----------------------------------------------------
    constexpr int32_t kMask[kCurrentVersion + 1][EStsElementLevel::kStsNofLevels] = {
      {(1 << kBits[0][0]) - 1, (1 << kBits[0][1]) - 1, (1 << kBits[0][2]) - 1,
       (1 << kBits[0][3]) - 1,
       (1 << kBits[0][4]) - 1, (1 << kBits[0][5]) - 1, (1 << kBits[0][6]) - 1},

      {(1 << kBits[1][0]) - 1, (1 << kBits[1][1]) - 1, (1 << kBits[1][2]) - 1,
       (1 << kBits[1][3]) - 1,
       (1 << kBits[1][4]) - 1, (1 << kBits[1][5]) - 1, (1 << kBits[1][6]) - 1}
    };

    XPU_D uint32_t GetVersion(int32_t address) {
        return uint32_t((address & (kVersionMask << kVersionShift)) >> kVersionShift);
    }

    XPU_D uint32_t GetElementId(int32_t address, int32_t level) {
        assert(level >= kStsSystem && level < kStsNofLevels);
        uint32_t version = GetVersion(address);

        return (address & (kMask[version][level] << kShift[version][level])) >> kShift[version][level];
    }

    struct AddressBucketsSmem {
        bool addressMap[148+6];
    };

    XPU_KERNEL(AddressBuckets, AddressBucketsSmem, const size_t n, const CbmStsDigiInput* digis, index_t* startIndex, index_t* endIndex, const index_t* splitIndex, size_t bucketCount, unsigned short splitCount, digi_t* output) {

        for (int i = 0; i < n; i += xpu::block_dim::x()) {
            const uint32_t address = digis[i].address;

            const unsigned short system = GetElementId(address, EStsElementLevel::kStsSystem);
            const unsigned short unit = GetElementId(address, EStsElementLevel::kStsUnit);
            const unsigned short ladder = GetElementId(address, EStsElementLevel::kStsLadder);
            const unsigned short hLadder = GetElementId(address, EStsElementLevel::kStsHalfLadder);
            const unsigned short module = GetElementId(address, EStsElementLevel::kStsModule);
            const unsigned short sensor = GetElementId(address, EStsElementLevel::kStsSensor);
            const unsigned short side = GetElementId(address, EStsElementLevel::kStsSide);

        }
        xpu::barrier();

    }
}