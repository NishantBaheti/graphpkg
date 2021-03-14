from matplotlib import style
import random
import datetime

## Importing the module
from graphpkg.live.graph import LiveTrend

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (8, 2)
})


style.use("dark_background")


def func1():
    return datetime.datetime.now(), [random.randrange(1, 1000), random.randrange(1, 1000)]


if __name__ == "__main__":
  
    trend1 = LiveTrend(func_for_data=func1, interval=1000,
                       title="plot 1 for range 1-100", window=500)
    trend1.start()

    plt.show()
