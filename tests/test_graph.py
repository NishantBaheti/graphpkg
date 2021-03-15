from graphpkg.live.graph import LiveTrend,LiveScatter,LiveDistribution
import random
import matplotlib.pyplot as plt

def test_trend_single():

    def get_new_data():
        return None, 1
    trend = LiveTrend(func_for_data = get_new_data, interval=1000)
    trend.start()

    plt.show()
    assert trend.ani is not None and trend.xs != [] and trend.ys != [[],[],[]]


def test_trend_multi():

    def get_new_data():
        return None, [1,2,3]
    trend = LiveTrend(func_for_data = get_new_data, interval=1000)
    trend.start()

    plt.show()
    assert trend.ani is not None and trend.xs != [] and trend.ys != [[],[],[]]


def test_trend_duplicate():

    def get_new_data():
        return 1, 1
    trend = LiveTrend(func_for_data = get_new_data, interval=1000)
    trend.start()

    plt.show()
    assert trend.ani is not None and trend.xs == [1] and trend.ys == [[1],[],[]]

def test_trend_boolean():

 
    def get_new_data():
        return 1, True
    trend = LiveTrend(func_for_data = get_new_data, interval=1000)
    trend.start()

    plt.show()

    assert trend.xs == [1] and trend.ys == [[1],[],[]]

    

def test_scatter1_single():

    def get_new_data():
        return random.randrange(1,10), random.randrange(1,10)
    scatter = LiveScatter(func_for_data = get_new_data, interval=1000)
    scatter.start()

    plt.show()

    assert scatter.ani is not None and scatter.xs != [] and scatter.ys != [[],[],[]]


def test_scatter_multi():

    def get_new_data():
        return random.randrange(1,10), [random.randrange(1,10),random.randrange(1,10),random.randrange(1,10)]
    scatter = LiveScatter(func_for_data = get_new_data, interval=1000)
    scatter.start()

    plt.show()

    assert scatter.ani is not None and scatter.xs != [] and scatter.ys != [[],[],[]]

def test_scatter_nothing():

    def get_new_data():
        return None, None
    scatter = LiveScatter(func_for_data = get_new_data, interval=1000)
    scatter.start()

    plt.show()

    assert scatter.ani is not None and scatter.xs == [] and scatter.ys == [[],[],[]]



def test_distribution_single():

    def get_new_data():
        return None, random.randrange(1,10)
    dist = LiveDistribution(func_for_data = get_new_data, interval=1000)
    dist.start()

    plt.show()

    assert dist.ani is not None and dist.xs != [] and dist.ys != [[],[],[]]

def test_distribution_multi():

    def get_new_data():
        return None, [random.randrange(1,10),random.randrange(1,10),random.randrange(1,10)]
    dist = LiveDistribution(func_for_data = get_new_data, interval=1000)
    dist.start()

    plt.show()

    assert dist.ani is not None and dist.xs != [] and dist.ys != [[],[],[]]
