
import random
import datetime
import matplotlib
#matplotlib.use('Agg')
from graphpkg.live import LiveDashboard
# plt.style.use('')

count1 = 0
cluster = 0.30

def func1():
    return datetime.datetime.now(), [random.randrange(1, 10) , random.randrange(1, 10) ]


def func2():
    return random.randrange(1, 100), [random.randrange(1, 10000) , random.randrange(1, 100), random.randrange(1, 100)]


def func3(*args):
    #print(args)
    return random.randrange(1, args[0]), [random.randrange(1, args[0]), random.randrange(1, 100)]       

if __name__ == "__main__":

    conf = {
        "dashboard": "DASHBOARD1",
        "plots": {
            "trend": [
                {
                    "func_for_data": func1,
                    "fig_spec": (3,3,(1,2)),
                    "interval": 500,
                    "title" : "trend plot1"
                },
                {
                    "func_for_data": func1,
                    "fig_spec": (3, 3, (4, 5)),
                    "interval" : 500,
                    "title" : "trend plot2"
                },
                {
                    "func_for_data": func1,
                    "fig_spec": (3, 3, (7, 8)),
                    "interval": 500,
                    "title": "trend plot3"
                },
            ],
            "scatter": [
                {
                    "fig_spec" : (3, 3, 3),
                    "func_for_data" : func3,
                    "func_args": (1000,),
                    "interval" : 1000,
                    "title" : "other scatter plot",
                    "window": 500
                },
                {
                    "fig_spec" : (3, 3, 6),
                    "func_for_data" : func2,
                    "interval": 500,
                    "title" : "some scatter plot",
                    "window": 1000
                },
                {
                    "fig_spec": (3, 3, 9),
                    "func_for_data": func3,
                    "func_args": (1000,),
                    "interval" : 1000,
                    "title" : "other other scatter plot",
                    "window": 500
                }
            ]
        }
    }
    
    dash = LiveDashboard(config=conf)
    dash.start()

    matplotlib.pyplot.show()


