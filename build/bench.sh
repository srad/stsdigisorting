#env bin/bash
rm -f benchmark_results.csv
rm -f benchmark_tp.csv
rm plots/*

stamp=$(date -d "today" +"%Y_%m_%d_%H_%M_%S")
runtime_file="runtime_$stamp.png"
speedup_file="speedup_ms_$stamp.png"
speedup_percent_file="speedup_percent_$stamp.png"
throughput_file="throughput_$stamp.png"

inc=1
r=$((inc))
DEVICE=cuda0
for i in {1..25}; do
  echo
  echo $i
  XPU_DEVICE=$DEVICE LD_LIBRARY_PATH=.:lib/xpu ./stsdigisort -i ../data/digis_2022-07-27_12-43-50.csv -r "$r"
  sleep 3s
  r=$((r + inc))
  XPU_DEVICE=$DEVICE python plot.py $runtime_file $speedup_file $speedup_percent_file $throughput_file
done
