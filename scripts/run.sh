#!/bin/bash

export CUDACXX=/usr/local/cuda/bin/nvcc
export hip_INCLUDE_DIR=/opt/rocm-5.4.2/include

rm output/*
rm benchmarks.sqlite
make -j64
XPU_DEVICE=cuda0 LD_LIBRARY_PATH=.:lib ./stsdigisort -i ../data/2/digis_2022-08-23_15-03-11_ev500_auau_12gev_mbias_1_2_1.csv -c -w #-r 1 -w -c
#XPU_DEVICE=cuda0 LD_LIBRARY_PATH=.:lib /usr/local/cuda-12.0/bin/compute-sanitizer --check-api-memory-access=yes --tool=memcheck --launch-timeout 1000 --target-processes all ./stsdigisort -i ../data/2/digis_2022-08-23_13-05-03_ev200_auau_25gev_centr_1_1_0.csv