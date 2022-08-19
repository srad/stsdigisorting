#pragma once

#include <xpu/device.h>

template<typename T>
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