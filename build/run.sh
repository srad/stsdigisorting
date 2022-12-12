rm output/*
rm benchmarks.sqlite
make -j64
XPU_DEVICE=cuda0 LD_LIBRARY_PATH=.:lib/xpu ./stsdigisort -i ../data/digis_2022-09-15_08-17-01-ev200_auau_25gev_centr_1-1-0.csv -r 1 -w -c
