import random
import datetime

## Importing the module
from graphpkg.live import LiveTrend

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize" : (15,3)
})



from matplotlib import style

style.use("dark_background")

def func1():
    return datetime.datetime.now(), random.randrange(1,100)

def func2():
    return None,random.randrange(1,1000)

if __name__ == "__main__":

    trend1 = LiveTrend(func_for_data=func1,interval=1000,title="plot 1 for range 1-100",window=500)
    trend1.start()

    trend2 = LiveTrend(func_for_data=func2,interval=1000,title="plot 2 for range 1-1000",window=100)
    trend2.start()

    plt.show()
