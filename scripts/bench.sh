#!/bin/bash

rm -fr plots/*
mkdir -p plots
rm benchmarks.sqlite

#    BlockSortBlockDimX=$thread_count
#    BlockSortItemsPerThread=$((2048 / JanSergeySortBlockDimX))

for digi_file in ../data/*.csv; do

  for device in cuda0 hip0; do
    # Handle different warp sizes.
    if [[ "$device" == cuda* ]]; then
      warp_size=32
    elif [[ "$device" == hip* ]]; then
      warp_size=64
    fi

    warp_multiplier=$((1024 / warp_size))

    # TODO: replace these static numbers by warp size increments, for device (32 nvidia, 64 amd)
    # but that will take the benchmark forever.
    for ((j=1;j<=warp_multiplier;j++)); do
      echo
      echo "=============================================================="
      echo "($j/$warp_multiplier) Running benchmark with WarpMultiplier=$j"
      echo "=============================================================="
      echo "JanSergeySortBlockDimX = $warp_size * $WarpMultiplier"
      echo
      JanSergeySortBlockDimX=$thread_count
      BlockSortBlockDimX=64
      BlockSortItemsPerThread=8

      echo "#pragma once

  namespace experimental {
      constexpr int channelCount = 2048;

      constexpr int WarpMultiplier = $j;

  #if XPU_IS_CUDA
      constexpr int WarpSize = 32;
  #else
      constexpr int WarpSize = 64;
  #endif

      constexpr int JanSergeySortBlockDimX = $warp_size * WarpMultiplier;
      constexpr int BlockSortBlockDimX = 64;
      constexpr int BlockSortItemsPerThread = 8;

      //static_assert((JanSergeySortBlockDimX % WarpSize) == 0, "Block dim X is not multiple of the warp size");
      //static_assert((BlockSortBlockDimX % WarpSize) == 0, "Block dim X is not multiple of the warp size");
  }" > ../src/constants.h

      if make -j64 ; then
        echo "Compiled."
      else
        echo "Compilation error."
        exit 1
      fi

      stamp=$(date -d "today" +"%Y_%m_%d_%H_%M_%S")

      folder="benchmark_${stamp}_${device}"
      mkdir -p "./plots/$folder"
      full_path="./plots/$folder"

      runtime_file="$folder/runtime.png"
      speedup_file="$folder/speedup_ms.png"
      speedup_percent_file="$folder/speedup_percent.png"
      throughput_file="$folder/throughput.png"

      DEVICE="$device"
      for r in {1..5}; do
        echo
        echo "Expand data by r=$r"
        XPU_DEVICE=$DEVICE LD_LIBRARY_PATH=.:lib ./stsdigisort -i "$digi_file" -r "$r" -b plots/$folder
        #sleep 3s
        # digis_2022-08-23_13-05-03_ev200_auau_25gev_centr_1_1_0.csv
        #XPU_DEVICE=$DEVICE FILENAME_RESULTS="$full_path/benchmark_results.csv" FILENAME_TP="$full_path/benchmark_tp.csv" python plot.py $runtime_file $speedup_file $speedup_percent_file $throughput_file
      done # for r
    done # for warp_multiplier
  done # for device

done # for digi_file