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

speedup = map(lambda x, y: max(float(x), 1.0) / max(float(y), 1.0), algo1, algo2)
  
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

plt.plot(n_in_millions, speedup, label = "Speedup", linestyle="-")

plt.xlabel("Digis (in millions)")
plt.ylabel("Speedup")
plt.savefig('./plots/' + sys.argv[2], dpi=300)