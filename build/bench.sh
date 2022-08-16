#env bin/bash
rm -f benchmark_results.csv
stamp=$(date -d "today" +"%Y_%m_%d_%H_%M_%S")
runtime_file="runtime_$stamp.png"
speedup_file="speedup_$stamp.png"
inc=1000000
n=$(inc)
DEVICE=cpu
for i in {1..100}; do
  echo
  echo $i
  XPU_DEVICE=$DEVICE LD_LIBRARY_PATH=.:lib/xpu ./stsdigisort -i ../data/digis_2022-07-27_12-43-50.csv -r 1000 -n "$n"
  n=$((n + inc))
  XPU_DEVICE=$DEVICE python plot.py $runtime_file $speedup_file
done
