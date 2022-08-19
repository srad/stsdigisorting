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

with open("benchmark_results.csv") as fp:
    reader = csv.reader(fp, delimiter=",")
    #next(reader, None)  # skip the headers
    headers = reader.next()
    data = [row for row in reader]


n_in_millions = map(lambda x:  int(x[0]) / 1000000, data)


#algo0 = np.array(map(lambda x: x[1], data), dtype=np.int)
algo1 = np.array(map(lambda x: x[1], data), dtype=np.float)
algo2 = np.array(map(lambda x: x[2], data), dtype=np.float)
algo3 = np.array(map(lambda x: x[3], data), dtype=np.float)

speedup = np.array(map(lambda x, y: float(x) - float(y), algo1, algo2))
  
# plot lines
#plt.plot(n_in_millions, algo0, label = headers[1], linestyle="-")
plt.plot(n_in_millions, algo1, label = headers[1], linestyle="-")
plt.plot(n_in_millions, algo2, label = headers[2], linestyle="--")
plt.plot(n_in_millions, algo3, label = headers[3], linestyle=":")

plt.title("Median runtime on device '" + os.getenv('XPU_DEVICE') + "'")
plt.xlabel("Digis (in millions)")
plt.ylabel("ms")

plt.legend()


if not os.path.exists("./plots"):
    os.makedirs("./plots")

plt.savefig('./plots/' + sys.argv[1], dpi=300)

plt.figure().clear()

fig, ax = plt.subplots()

z = np.array([1.0] * len(n_in_millions))
plt.plot(n_in_millions, speedup, linestyle="-")
plt.fill_between(n_in_millions, speedup, 1.0, where=(speedup > z), alpha=0.20, facecolor="green", interpolate=True)
plt.fill_between(n_in_millions, speedup, 1.0, where=(speedup < z), alpha=0.20, facecolor="red", interpolate=True)
plt.axhline(y=1.0, color="green", linestyle="--")

plt.title("Speedup JanSergeySort vs. BlockSort (median: " + ('%.2f' % np.median(speedup)) + "ms)")
plt.xlabel("digis (in millions)")
plt.ylabel("speedup (ms)")
plt.savefig('./plots/' + sys.argv[2], dpi=300)