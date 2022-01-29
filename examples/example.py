
import matplotlib.pyplot as plt
from matplotlib import style
import random
import datetime
from graphpkg.live import LiveTrend,LiveScatter

style.use("dark_background")


def get_new_data():
    return datetime.datetime.now(), [random.randrange(5, 10),random.randrange(1,5)]

# def get_new_data1():
#     y_data = random.randrange(0, 10)
#     return None, y_data if y_data > 5 else None


def func2():
    return random.randrange(1, 100), [random.randrange(1, 100), random.randrange(1, 100), random.randrange(1, 100)]


def func3(*args):
    return random.randrange(1, args[0]), [random.randrange(1, args[0]), random.randrange(1, 100)]

if __name__ == "__main__":
    lg1 = LiveTrend(
        func_for_data=get_new_data,
        interval=1000,
        title="Live trend with date time"
    )
    lg1.start()

    
    g2 = LiveScatter(func_for_data=func2, interval=1000,
                     title="scatter with 3 plots", window=1000)
    g2.start()

    g3 = LiveScatter(func_for_data=func3, func_args=(
        500,), interval=1000, title="scatter with 2 plots", window=500)
    g3.start()

    plt.show()
