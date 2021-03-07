
import matplotlib.pyplot as plt
from matplotlib import style
import random
import datetime
from graphpkg.live_graph import LiveTrend

style.use("dark_background")


def get_new_data():
    return datetime.datetime.now(), random.randrange(0, 10)

def get_new_data1():
    y_data = random.randrange(0, 10)
    return None, y_data if y_data > 5 else None


if __name__ == "__main__":
    lg1 = LiveTrend(
        func_for_data=get_new_data,
        interval=1000
    )
    lg1.start()

    lg2 = LiveTrend(
        func_for_data=get_new_data1,
        interval=1000, title="my test data"
    )
    lg2.start()
    plt.show()
