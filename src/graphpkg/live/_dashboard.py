"""
Devloped By : Nishant Baheti

"""
from typing import List,Dict
import matplotlib.pyplot
from graphpkg.live import LiveTrend, LiveScatter, LiveDistribution
from graphpkg import __version__


__author__ = "Nishant Baheti"
__copyright__ = "Nishant Baheti"
__license__ = "MIT"

__plot_class_map__ = {
    "trend": LiveTrend,
    "scatter": LiveScatter,
    "distribution": LiveDistribution
}

class LiveDashboard:
    """Live Dashboard plot

    Args:
        config (dict): Configuration Dictionary

    Example:
        >>> conf = {
        >>>     "dashboard": "DASHBOARD1",
        >>>     "plots": {
        >>>         "trend": [
        >>>             {
        >>>                 "func_for_data": func1,
        >>>                 "fig_spec": (4, 3, (1, 2)),
        >>>                 "interval": 500,
        >>>                 "title": "trend plot1"
        >>>             },
        >>>             {
        >>>                 "func_for_data": func1,
        >>>                 "fig_spec": (4, 3, (4, 5)),
        >>>                 "interval": 500,
        >>>                 "title": "trend plot2"
        >>>             },
        >>>             {
        >>>                 "func_for_data": func1,
        >>>                 "fig_spec": (4, 3, (7, 8)),
        >>>                 "interval": 500,
        >>>                 "title": "trend plot3"
        >>>             },
        >>>             {
        >>>                 "func_for_data": func1,
        >>>                 "fig_spec": (4, 3, (10,11)),
        >>>                 "interval": 500,
        >>>                 "title": "trend plot4"
        >>>             }
        >>>         ],
        >>>         "distribution": [{
        >>>             "fig_spec": (4, 3, (3,6)),
        >>>             "func_for_data": func4,
        >>>             "interval": 1000,
        >>>             "title": "distribution plot",
        >>>             "window": 500
        >>>         }],
        >>>         "scatter": [
        >>>             {
        >>>                 "fig_spec": (4, 3, (9,12)),
        >>>                 "func_for_data": func3,
        >>>                 "func_args": (1000,),
        >>>                 "interval": 1000,
        >>>                 "title": "other other scatter plot",
        >>>                 "window": 500
        >>>             }
        >>>         ]
        >>>     }
        >>> }
        >>> dash = LiveDashboard(config=conf)
        >>> dash.start()
        >>> matplotlib.pyplot.show()

    """

    def __init__(self, config: dict):
        """Constructor
        """
        self.config = config
        self._dash_config = []
        self.fig = matplotlib.pyplot.figure()
        self.title = None

    @property
    def dash_config(self)-> List[Dict]:
        """Dash board configuration

        Returns:
            List[Dict]: list of plots in a dictionary 
        """
        return self._dash_config

    def _plot_config(self):
        """plot config template

        Returns:
            dict : Configuration template dictionary
        """

        conf = {
            "interval": 1000,
            "func_for_data": None,
            "func_args": None,
            "fig": self.fig,
            "fig_spec": None,
            "xlabel": "x-axis",
            "ylabel": "y-axis",
            "label": "Current Data",
            "title": "Live Trend",
            "window": 100
        }
        return conf.copy()

    def _load_config_in_format(self):
        """Load configuratiobn of the dashboard

        Raises:
            ValueError: any of these : func_for_data, interval, fig_spec, fig are None
        """
        self.title = self.config["dashboard"] or "DASHBOARD"
        plots_in_conf = self.config["plots"].keys()
        for plot in plots_in_conf:
            if plot in __plot_class_map__.keys():
                plot_class = __plot_class_map__[plot]
                list_of_plots = self.config["plots"][plot]

                for cplot in list_of_plots:

                    templ_conf = self._plot_config()

                    for key in templ_conf:
                        if key in cplot:
                            templ_conf[key] = cplot[key]

                    if None in [
                        templ_conf["func_for_data"],
                        templ_conf["interval"],
                        templ_conf["fig_spec"],
                        templ_conf["fig"]
                    ]:
                        raise ValueError(
                            """ func_for_data, interval, fig_spec, fig can't be None""")
                    else:
                        self._dash_config.append(plot_class(**templ_conf))

    def start(self):
        """Start the dashboard
        """
        self._load_config_in_format()

        for plot in self._dash_config:
            plot.start()

        self.fig.canvas.manager.set_window_title(self.title)
        self.fig.tight_layout()
        

    def display(self):
        """display information
        """
        print(f"""
        =====================================================
        Configuration Information
        =====================================================

        title   : {self.title}
        
        """)
