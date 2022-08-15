#env bin/bash
rm -f benchmark_results.csv
#echo "n,algo,min,max,median\n" > benchmark_results.csv
inc=1000000
n=$(inc)
for i in {1..100}; do
  echo $i;
  XPU_DEVICE=hip0 LD_LIBRARY_PATH=.:lib/xpu ./stsdigisort -i ../data/digis_2022-07-27_12-43-50.csv -r 1000 -n $n
  n=$((n + inc))
done