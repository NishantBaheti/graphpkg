
import random
import datetime
import matplotlib
from graphpkg.live import LiveDashboard
# matplotlib.pyplot.style.use("seaborn")
matplotlib.pyplot.rcParams.update({
    'legend.fontsize': 6,
    'legend.handlelength': 2
})

count1 = 0
cluster = 0.30
mu = 60
sigma = 30


def func1():
    return datetime.datetime.now(), [random.randrange(1, 10), random.randrange(1, 10)]


def func2():
    return random.randrange(1, 100), [random.randrange(1, 10000), random.randrange(1, 100), random.randrange(1, 100)]


def func3(*args):
    # print(args)
    return random.randrange(1, args[0]), [random.randrange(1, args[0]), random.randrange(1, 100)]


def func4():
    return None, mu + sigma *random.randrange(-100, 100)


if __name__ == "__main__":

    conf = {
        "dashboard": "DASHBOARD1",
        "plots": {
            "trend": [
                {
                    "func_for_data": func1,
                    "fig_spec": (4, 3, (1, 2)),
                    "interval": 500,
                    "title": "trend plot1"
                },
                {
                    "func_for_data": func1,
                    "fig_spec": (4, 3, (4, 5)),
                    "interval": 500,
                    "title": "trend plot2"
                },
                {
                    "func_for_data": func1,
                    "fig_spec": (4, 3, (7, 8)),
                    "interval": 500,
                    "title": "trend plot3"
                },
                {
                    "func_for_data": func1,
                    "fig_spec": (4, 3, (10,11)),
                    "interval": 500,
                    "title": "trend plot4"
                }
            ],
            "distribution": [{
                "fig_spec": (4, 3, (3,6)),
                "func_for_data": func4,
                "interval": 500,
                "title": "distribution plot",
                "window": 5000
            }],
            "scatter": [
                {
                    "fig_spec": (4, 3, (9,12)),
                    "func_for_data": func3,
                    "func_args": (1000,),
                    "interval": 500,
                    "title": "other other scatter plot",
                    "window": 500
                }
            ]
        }
    }

    dash = LiveDashboard(config=conf)
    dash.start()

    matplotlib.pyplot.show()
