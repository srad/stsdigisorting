make -j64
XPU_DEVICE=hip0 LD_LIBRARY_PATH=.:lib/xpu ./stsdigisort -i ../data/digis_2022-08-23_13-05-03_ev200_auau_25gev_centr_1_1_0.csv -r 1 -w -c
