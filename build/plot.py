from email import header
import matplotlib
matplotlib.use('Agg')
import csv

import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import os
import sys

filename_results = os.getenv('FILENAME_RESULTS', 'benchmark_results.csv')
filename_tp = os.getenv('FILENAME_TP', 'benchmark_tp.csv')

with open(filename_results) as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    #next(reader, None)  # skip the headers
    headers = reader.next()
    data = [row for row in reader]

n_in_millions = map(lambda x:  int(x[0]) / 100000, data)

algo = []
n = len(headers)

for i in range(1, n):
    algo.append(np.array(map(lambda x: x[i], data), dtype=np.float))

speedup = np.array(map(lambda x, y: float(x) - float(y), algo[0], algo[1]))
speedup_percent = np.array(map(lambda x, y: (float(x) - float(y)) / float(x) * 100, algo[0], algo[1]))

for i in range(1, n):
    plt.plot(n_in_millions, algo[i-1], label = headers[i])

# ------------------------------------------------------------------------------------------
# 1. Runtime 
# ------------------------------------------------------------------------------------------

plt.title("Median runtime on device '" + os.getenv('XPU_DEVICE') + "'")
plt.xlabel("Digis (10^5)")
plt.ylabel("ms")
plt.grid(linestyle='dotted')

plt.legend()

if not os.path.exists("./plots"):
    os.makedirs("./plots")

plt.savefig('./plots/' + sys.argv[1], dpi=300)

# ------------------------------------------------------------------------------------------
# 2. Speedup ms
# ------------------------------------------------------------------------------------------

plt.figure().clear()

fig, ax = plt.subplots()

z = np.array([0.0] * len(n_in_millions))
plt.plot(n_in_millions, speedup, linestyle="-")
plt.fill_between(n_in_millions, speedup, 0.0, where=(speedup > z), alpha=0.20, facecolor="green", interpolate=True)
plt.fill_between(n_in_millions, speedup, 0.0, where=(speedup < z), alpha=0.20, facecolor="red", interpolate=True)
plt.axhline(y=0.0, color="green", linestyle="--")

plt.title("Speedup JanSergeySort vs. BlockSort (median: " + ('%.2f' % np.median(speedup)) + "ms)")
plt.xlabel("digis (10^5)")
plt.ylabel("speedup (ms)")
plt.grid(linestyle='dotted')
plt.savefig('./plots/' + sys.argv[2], dpi=300)

# ------------------------------------------------------------------------------------------
# 3. Speedup %
# ------------------------------------------------------------------------------------------

plt.figure().clear()

fig, ax = plt.subplots()

z = np.array([0.0] * len(n_in_millions))
plt.plot(n_in_millions, speedup_percent, linestyle="-")
plt.fill_between(n_in_millions, speedup_percent, 0.0, where=(speedup_percent > z), alpha=0.20, facecolor="green", interpolate=True)
plt.fill_between(n_in_millions, speedup_percent, 0.0, where=(speedup_percent < z), alpha=0.20, facecolor="red", interpolate=True)
plt.axhline(y=0.0, color="green", linestyle="--")
plt.grid(linestyle='dotted')

plt.title("Speedup JanSergeySort vs. BlockSort (median: " + ('%.2f' % np.median(speedup_percent)) + "%)")
plt.xlabel("digis (10^5)")
plt.ylabel("speedup (%)")
plt.savefig('./plots/' + sys.argv[3], dpi=300)

# ------------------------------------------------------------------------------------------
# 4. Throughput
# ------------------------------------------------------------------------------------------

plt.figure().clear()

with open(filename_tp) as fp:
    reader = csv.reader(fp, delimiter=",")
    #next(reader, None)  # skip the headers
    headers_tp = reader.next()
    data_tp = [row for row in reader]

algo = []
n = len(headers_tp)

for i in range(1, n):
    algo.append(np.array(map(lambda x: x[i], data_tp), dtype=np.float))

speedup = np.array(map(lambda x, y: x - y, algo[0], algo[1]))

for i in range(1, n):
    plt.plot(n_in_millions, algo[i-1], label = headers_tp[i])

plt.title("Median throughput on device '" + os.getenv('XPU_DEVICE') + "'")
plt.xlabel("Digis (10^5)")
plt.ylabel("GB/s")
plt.grid(linestyle='dotted')
plt.legend()

plt.savefig('./plots/' + sys.argv[4], dpi=300)