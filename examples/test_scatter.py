import random
import datetime

## Importing the module
from graphpkg.live import LiveScatter

import matplotlib.pyplot as plt

from matplotlib import style

style.use("dark_background")

def func1():
    return random.randrange(1,100), random.randrange(1,100)

def func2():
    return random.randrange(1,100), [ random.randrange(1,100),random.randrange(1,100),random.randrange(1,100)]

def func3(*args):
    return random.randrange(1,args[0]), [ random.randrange(1,args[0]),random.randrange(1,100)]


if __name__ == "__main__":

    g1 = LiveScatter(func_for_data=func1,interval=500,title="plot 1 for range 1-100",window=500)
    g1.start()

    g2 = LiveScatter(func_for_data=func2,interval=500,title="plot 2 for range 1-100",window=1000)
    g2.start()

    g3 = LiveScatter(func_for_data=func3,func_args=(500,),interval=1000,title="plot 1 for range 1-100",window=500)
    g3.start()

    plt.show()
