make -j64
XPU_DEVICE=cuda0 LD_LIBRARY_PATH=.:lib/xpu ./stsdigisort -i ../data/digis_2022-07-27_12-43-50.csv -r 1 -w
