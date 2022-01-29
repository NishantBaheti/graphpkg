import datetime 
import random
import matplotlib.pyplot as plt

from graphpkg.live import LiveTrend,LiveScatter
# plt.style.use('')

count1 = 0

cluster = 0.30

def func1():
    global count1
    count1 += 1
    return datetime.datetime.now(), [random.randrange(1, 10) + count1,random.randrange(1,10)+ count1]

def func2():
    global cluster
    return random.randrange(1, 100), [random.randrange(1, 10000) * cluster, random.randrange(1, 100), random.randrange(1, 100)]

def func3(*args):
    return random.randrange(1, args[0]), [random.randrange(1, args[0]), random.randrange(1, 100)]


if __name__ == "__main__":

    fig = plt.figure()
    
    trend1 = LiveTrend(
        fig=fig,
        fig_spec=(3,3,(1,2)),
        func_for_data=func1,
        interval=500,
        title="trend plot"
    )
    trend1.start()

    trend2 = LiveTrend(
        fig=fig,
        fig_spec=(3, 3, (4, 5)),
        func_for_data=func1,
        interval=500,
        title="other trend plot"
    )
    trend2.start()

    trend3 = LiveTrend(
        fig=fig,
        fig_spec=(3, 3, (7, 8)),
        func_for_data=func1,
        interval=500,
        title="other other trend plot"
    )
    trend3.start()

    scatter1 = LiveScatter(
        fig = fig,
        fig_spec=(3,3,3),
        func_for_data=func2, 
        interval=500,
        title="some scatter plot", 
        window=1000
    )
    scatter1.start()

    scatter2 = LiveScatter(
        fig=fig,
        fig_spec=(3, 3, 6),
        func_for_data=func3, 
        func_args=(1000,), 
        interval=1000, 
        title="other scatter plot", 
        window=500
    )
    scatter2.start()
    
    scatter3 = LiveScatter(
        fig=fig,
        fig_spec=(3, 3, 9),
        func_for_data=func3, 
        func_args=(1000,), 
        interval=1000, 
        title="other other scatter plot", 
        window=500
    )
    scatter3.start()    


    fig.canvas.set_window_title("dashboard")
    plt.show()
