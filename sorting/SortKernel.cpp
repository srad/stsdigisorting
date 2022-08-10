#include "SortKernel.h"
#include "../datastructures.h"

// Initialize the xpu image. This macro must be placed once somewhere in the device sources.
XPU_IMAGE(BlockSortKernel);

#if XPU_IS_CUDA
#define CBM_STS_SORT_BLOCK_SIZE 32
#define CBM_STS_SORT_ITEMS_PER_THREAD 32
#else
#define CBM_STS_SORT_BLOCK_SIZE 256
#define CBM_STS_SORT_ITEMS_PER_THREAD 6
#endif

// Optional shorthand for the sorting class.
//
// Template arguments are the type of the key that is sorted,
// size of the gpu block (currently hard-coded at 64 threads)
// and the number of keys that are sorted by each thread with
// the underlying cub::BlockRadixSort implementation.
using SortT = xpu::block_sort<unsigned long int, experimental::CbmStsDigi, CBM_STS_SORT_BLOCK_SIZE, CBM_STS_SORT_ITEMS_PER_THREAD>;

// Define type that is used to allocate shared memory.
// In this case only shared memory for the underlying cub::BlockRadixSort is needed.
struct GpuSortSmem {
    typename SortT::storage_t sortBuf;
};

XPU_KERNEL(BlockSort, GpuSortSmem, experimental::CbmStsDigi* data, experimental::CbmStsDigi* buf, experimental::CbmStsDigi** out, const size_t numElems) {
    // Call the sort function. Along the two buffers and the number of elements, a function that
    // extracts the key from the struct has to be passed.
    // Returns the buffer that contains the sorted data (either data or buf).

    const size_t itemsPerBlock = numElems / xpu::grid_dim::x();
    const size_t offset = itemsPerBlock * xpu::block_idx::x();

    experimental::CbmStsDigi* res = SortT(smem.sortBuf).sort(
        &data[offset], itemsPerBlock, &buf[offset],
        [](const experimental::CbmStsDigi& a) { return ((unsigned long int) a.channel) << 32 | (unsigned long int) (a.time); }
    );

    if (xpu::block_idx::x() == 0) {
        *out = res;
    }
}

/*
  xpu::run_kernel<SortDigis>(xpu::grid::n_blocks(hfc.nModules * 2));
XPU_KERNEL(SortDigis, CbmStsSortDigiSmem) { xpu::cmem<HitFinder>().sortDigisInSpaceAndTime(smem, xpu::block_idx::x()); }

using SortDigisT = xpu::block_sort<unsigned long int, CbmStsDigi, xpu::block_size<SortDigis> {}, CBM_STS_SORT_ITEMS_PER_THREAD>;

struct CbmStsSortDigiSmem {
    // TODO no magic numbers for blocksize
    typename SortDigisT::storage_t sortBuf;
};

XPU_D void CbmStsGpuHitFinder::sortDigisInSpaceAndTime(CbmStsSortDigiSmem& smem, int iBlock) const
{
  int iModule          = iBlock;
  CbmStsDigi* digis    = &digisPerModule[digiOffsetPerModule[iModule]];
  CbmStsDigi* digisTmp = &digisPerModuleTmp[digiOffsetPerModule[iModule]];
  int nDigis           = getNDigis(iModule);

  SortDigisT digiSort(smem.sortBuf);

  digis = digiSort.sort(digis, nDigis, digisTmp, [](const CbmStsDigi a) {
    return ((unsigned long int) a.fChannel) << 32 | (unsigned long int) (a.fTime);
  });

  if (xpu::thread_idx::x() == 0) { digisSortedPerModule[iModule] = digis; }
}


XPU_KERNEL(SortDigis, CbmStsSortDigiSmem) { xpu::cmem<HitFinder>().sortDigisInSpaceAndTime(smem, xpu::block_idx::x()); }
*/
