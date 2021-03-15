from graphpkg.live.dashboard import LiveDashboard
import random
import matplotlib.pyplot as plt


def test_dashboard_all_graphs():

    def func1():
        return None, 1

    def func2():
        return None, random.randrange(1,10)

    def func3():
        return random.randrange(1,10), random.randrange(1,10)

    conf = {
        "dashboard": "DASHBOARD1",
        "plots": {
            "trend": [
                {
                    "func_for_data": func1,
                    "fig_spec": (2, 2, (1, 2)),
                    "interval": 500,
                    "title": "trend plot"
                }
            ],
            "distribution": [
                {
                "fig_spec": (2, 2, 3),
                "func_for_data": func2,
                "interval": 1000,
                "title": "distribution plot",
                "window": 500
                }
            ],
            "scatter": [
                {
                    "fig_spec": (2, 2, 4),
                    "func_for_data": func3,
                    "interval": 1000,
                    "title": "scatter plot",
                    "window": 500
                }
            ]
        }
    }
    dash = LiveDashboard(config=conf)
    dash.start()

    plt.plot()

    assert dash.dash_config != []

test_dashboard_all_graphs()