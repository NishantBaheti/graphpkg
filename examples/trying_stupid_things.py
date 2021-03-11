import matplotlib.pyplot as plt 
import numpy as np 
from scipy import stats
import random

data = np.random.rand(2)
# data = [random.randint(1,1000) for i in range(1000)]
kernel = stats.gaussian_kde(data)


fig = plt.figure()
ax = fig.add_subplot(111)
count, bins, ignored = ax.hist(data, 30, density=True,alpha=0.4)
ax.plot(bins,kernel(bins))
plt.show()
