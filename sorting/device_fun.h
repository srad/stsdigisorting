#pragma once

#include <xpu/device.h>

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