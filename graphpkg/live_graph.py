"""
=======================================================================================================
    
    Devloped By : Nishant Baheti

=======================================================================================================
"""

from typing import Any, Callable, Iterable
import matplotlib.pyplot
import matplotlib.animation 

class LiveTrend:
    """Live Trend Graph Module

    Args:
        func_for_data (callable): Function to return x and y data points.x is a single value and y can be a
                                    list of max length 3 or a single value.
        
        Example:
            >>> def get_new_data():
            >>>     return datetime.datetime.now(),10

            >>> def get_new_data():
            >>>     return None,10

            >>> def get_new_data():
            >>>     ## first param for x axis and second can be an array of values
            >>>     return datetime.datetime.now(),[10,11]

            >>> def func1(*args):
            >>>    return datetime.datetime.now(),random.randrange(1, args[0])

        func_args (Iterable, optional): data function arguments. Defaults to None.
        fig (matplotlib.pyplot.figure, optional): [description]. Defaults to None.
        fig_spec (tuple, optional): [description]. Defaults to (1,1,1).
        interval (int): Interval to refresh data in milliseconds.
        xlabel (str, optional): Label for x-axis. Defaults to "x-axis".
        ylabel (str, optional): Label for y-axis. Defaults to "y-axis".
        label (str, optional): Label for plot line. Defaults to "Current Data".
        title (str, optional): Title of trend chart. Defaults to "Live Trend".
        window (int, optional): Data point window. Defaults to 50.

    Examples:
        >>> trend = LiveTrend(func_for_data = get_new_data, interval=1000)
        >>> trend.start()
        >>> matplotlib.pyplot.show()

        >>> trend = LiveTrend(func_for_data = get_new_data, interval=1000, window=30)
        >>> trend.start()
        >>> matplotlib.pyplot.show()

        >>> trend = LiveTrend(func_for_data = get_new_data, interval=1000, title="my test data")
        >>> trend.start()
        >>> matplotlib.pyplot.show()
    """

    def __init__(
        self, 
        interval: int, 
        func_for_data: callable,
        func_args: Iterable = None,
        fig: matplotlib.pyplot.figure = None,
        fig_spec: tuple = (1,2,(1,2)), 
        xlabel: str = "x-axis", 
        ylabel: str = "y-axis", 
        label: str = "Current Data", 
        title: str = "Live Trend", 
        window: int = 50) -> None:
        """[summary]
        """
        self.func_for_data = func_for_data
        self.func_args = func_args
        self.interval = int(interval)
        self.xlabel = str(xlabel)
        self.ylabel = str(ylabel)
        self.label = str(label)
        self.title = str(title)
        self.window = int(window)
        self.xs = []
        self._max_line_plots = 3
        self.ys = [[] for i in range(self._max_line_plots)]
        self.fig = fig or matplotlib.pyplot.figure()
        self.fig.canvas.set_window_title(self.title)
        self.ax = self.fig.add_subplot(*fig_spec)
        self.fig.tight_layout()
        self.ani = None
        self.counter = 0

    def _plot_single_line(self) -> None:
        """Plot single line in trend chart
        """
        self.ax.clear()
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.plot(self.xs, self.ys[0], marker = "o", markersize= 0.75, linewidth = 0.6, label=self.label)
        self.ax.grid(color='grey', linewidth=0.3, visible=True)
        self.ax.legend(loc="upper left")
        
    def _plot_multi_line(self,num_of_plots: int) -> None:
        """Plot multi line in trend chart

        Args:
            num_of_plots (int): Number of plots
        """
        self.ax.clear()
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        for p_i in range(num_of_plots):
            self.ax.plot(
                self.xs, self.ys[p_i], marker = "o", markersize= 0.75,linewidth = 0.6, label=self.label+"-"+str(p_i+1))
        self.ax.grid(color='grey', linewidth=0.3, visible=True)
        self.ax.legend(loc="upper left")

    def _animate(self,i:int) -> None:
        """Animation method to update chart

        Args:
            i (int): counter
        """
        self.counter = i
        if self.func_args is not None: 
            x_data, y_data = self.func_for_data(*self.func_args)
        else:
            x_data, y_data = self.func_for_data()


        x_data = x_data or self.counter
        plot_chart = True
        if len(self.xs) > 0:
            if self.xs[-1] == x_data:
                plot_chart = False

        if plot_chart:
            if isinstance(y_data, float) or isinstance(y_data, int) or isinstance(y_data, str):
                self.xs.append(x_data)
                self.ys[0].append(y_data)
                self.xs = self.xs[-self.window:]
                self.ys = self.ys[-self.window:]
                self._plot_single_line()

            elif isinstance(y_data,list):
                
                self.xs.append(x_data)
                self.xs = self.xs[-self.window:]

                num_of_plots = self._max_line_plots if len(
                    y_data) > self._max_line_plots else len(y_data)
                for i in range(num_of_plots):
                    self.ys[i].append(y_data[i])
                    self.ys[i] = self.ys[i][-self.window:]
                
                self._plot_multi_line(num_of_plots)            

            else:
                raise ValueError("y-axis datatype is not accepted.")
        else:
            pass
        
    def start(self) -> None:
        """Initiate the trend chart
        """
        self.ani = matplotlib.animation.FuncAnimation(
            self.fig, 
            self._animate,
            interval=self.interval
        )

    def display(self) -> None:
        """display information
        """
        print(f"""
        =====================================================
        Configuration Information
        =====================================================

        func_for_data   : {self.func_for_data} 
        interval        : {self.interval}
        xlabel          : {self.xlabel} 
        ylabel          : {self.ylabel}  
        label           : {self.label}
        title           : {self.title}  
        window          : {self.window}
        
        """)

class LiveScatter:
    """Live Scatter Graph Module

    Args:
        func_for_data (callable): Function to return x and y data point.x is a single value and y can be a
                                    list of max length 3 or a single value. both of them shouldn't be None.
        
        Example:
            >>> def get_new_data():
            >>>     return 10,10

            >>> def get_new_data():
            >>>     ## first param for x axis and second can be an array of values
            >>>     return 10,[10,11]

            >>> def func1(*args):
            >>>    return random.randrange(1, args[0]),random.randrange(1, args[0])

        func_args (Iterable, optional): data function arguments. Defaults to None.
        fig (matplotlib.pyplot.figure, optional): [description]. Defaults to None.
        fig_spec (tuple, optional): [description]. Defaults to (1,1,1).
        interval (int): Interval to refresh data in milliseconds.
        xlabel (str, optional): Label for x-axis. Defaults to "x-axis".
        ylabel (str, optional): Label for y-axis. Defaults to "y-axis".
        label (str, optional): Label for plot line. Defaults to "Current Data".
        title (str, optional): Title of Scatter chart. Defaults to "Live Scatter".
        window (int, optional): Data point window. Defaults to 500.

    Examples:
        >>> scatter = LiveScatter(func_for_data = get_new_data, interval=1000)
        >>> scatter.start()
        >>> matplotlib.pyplot.show()

        >>> scatter = LiveScatter(func_for_data = get_new_data, interval=1000, window=30)
        >>> scatter.start()
        >>> matplotlib.pyplot.show()

        >>> scatter = LiveScatter(func_for_data = get_new_data,func_args=(1000,), interval=1000, title="my test data")
        >>> scatter.start()
        >>> matplotlib.pyplot.show()
    """

    def __init__(
        self, 
        interval: int, 
        func_for_data: callable,
        func_args: Iterable = None,
        fig: matplotlib.pyplot.figure = None,
        fig_spec: tuple = (1, 1, 1),
        xlabel: str = "x-axis", 
        ylabel: str = "y-axis", 
        label: str = "Current Data", 
        title: str = "Live Scatter", 
        window: int = 500) -> None:
        """[summary]
        """
        self.func_for_data = func_for_data
        self.func_args = func_args
        self.interval = int(interval)
        self.xlabel = str(xlabel)
        self.ylabel = str(ylabel)
        self.label = str(label)
        self.title = str(title)
        self.window = int(window)
        self.xs = []
        self._max_scatter_plots = 3
        self.ys = [[] for i in range(self._max_scatter_plots)]
        self.fig = fig or matplotlib.pyplot.figure()
        self.fig.canvas.set_window_title(self.title)
        self.ax = self.fig.add_subplot(*fig_spec)
        self.fig.tight_layout()
        self.ani = None
        self.counter = 0

    def _plot_single_scatter(self) -> None:
        """Plot single line in trend chart
        """
        self.ax.clear()
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.scatter(self.xs, self.ys[0], s=[10], alpha= 0.6, label=self.label)
        self.ax.grid(color='grey', linewidth=0.3, visible=True)
        self.ax.legend(loc="upper left")
        
    def _plot_multi_scatter(self,num_of_plots: int) -> None:
        """Plot multi scatter chart

        Args:
            num_of_plots (int): Number of plots
        """
        self.ax.clear()
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        for p_i in range(num_of_plots):
            self.ax.scatter(
                self.xs, self.ys[p_i], s=[10], alpha= 0.6, label=self.label+"-"+str(p_i+1))
        self.ax.grid(color='grey', linewidth=0.3, visible=True)
        self.ax.legend(loc="upper left")

    def _animate(self,i:int) -> None:
        """Animation method to update chart

        Args:
            i (int): counter
        """
        self.counter = i
        if self.func_args is not None: 
            x_data, y_data = self.func_for_data(*self.func_args)
        else:
            x_data, y_data = self.func_for_data()


        if None not in [x_data,y_data]:
            if isinstance(y_data, float) or isinstance(y_data, int) or isinstance(y_data, str):
                self.xs.append(x_data)
                self.ys[0].append(y_data)
                self.xs = self.xs[-self.window:]
                self.ys = self.ys[-self.window:]
                self._plot_single_scatter()

            elif isinstance(y_data,list):
                
                self.xs.append(x_data)
                self.xs = self.xs[-self.window:]

                num_of_plots = self._max_scatter_plots if len(
                    y_data) > self._max_scatter_plots else len(y_data)
                for i in range(num_of_plots):
                    self.ys[i].append(y_data[i])
                    self.ys[i] = self.ys[i][-self.window:]
                
                self._plot_multi_scatter(num_of_plots)            

            else:
                raise ValueError("y-axis datatype is not accepted.")

    def start(self) -> None:
        """Initiate the scatter chart
        """
        self.ani = matplotlib.animation.FuncAnimation(
            self.fig, 
            self._animate,
            interval=self.interval
        )

    def display(self) -> None:
        """display information
        """
        print(f"""
        =====================================================
        Configuration Information
        =====================================================

        func_for_data   : {self.func_for_data} 
        interval        : {self.interval}
        xlabel          : {self.xlabel} 
        ylabel          : {self.ylabel}  
        label           : {self.label}
        title           : {self.title}  
        window          : {self.window}
        
        """)


__plot_class_map__ = {
    "trend": LiveTrend,
    "scatter": LiveScatter
}


class LiveDashboard:
    """Live Dashboard plot

    Args:
        config (dict): Configuration Dictionary

    Notes:

        Dashboard Configration Example

            |    conf = {
            |        "dashboard": "DASHBOARD1",
            |        "plots": {
            |            "trend": [
            |                {
            |                    "func_for_data": func1,
            |                    "fig_spec": (1,3,(1,2)),
            |                    "interval": 500,
            |                    "title" : "trend plot1"
            |                }
            |            ],
            |            "scatter": [
            |                {
            |                    "fig_spec" : (1, 3, 3),
            |                    "func_for_data" : func3,
            |                    "func_args": (1000,),
            |                    "interval" : 1000,
            |                    "title" : "other scatter plot",
            |                    "window": 500
            |                }
            |            ]
            |        }
            |    }

    Example:
        >>> conf = {
        >>>     "dashboard": "DASHBOARD1",
        >>>     "plots": {
        >>>         "trend": [
        >>>             {
        >>>                 "func_for_data": func1,
        >>>                 "fig_spec": (3,3,(1,2)),
        >>>                 "interval": 500,
        >>>                 "title" : "trend plot1"
        >>>             },
        >>>             {
        >>>                 "func_for_data": func1,
        >>>                 "fig_spec": (3, 3, (4, 5)),
        >>>                 "interval" : 500,
        >>>                 "title" : "trend plot2"
        >>>             },
        >>>             {
        >>>                 "func_for_data": func1,
        >>>                 "fig_spec": (3, 3, (7, 8)),
        >>>                 "interval": 500,
        >>>                 "title": "trend plot3"
        >>>             },
        >>>         ],
        >>>         "scatter": [
        >>>             {
        >>>                 "fig_spec" : (3, 3, 3),
        >>>                 "func_for_data" : func3,
        >>>                 "func_args": (1000,),
        >>>                 "interval" : 1000,
        >>>                 "title" : "other scatter plot",
        >>>                 "window": 500
        >>>             },
        >>>             {
        >>>                 "fig_spec" : (3, 3, 6),
        >>>                 "func_for_data" : func2,
        >>>                 "interval": 500,
        >>>                 "title" : "some scatter plot",
        >>>                 "window": 1000
        >>>             },
        >>>             {
        >>>                 "fig_spec": (3, 3, 9),
        >>>                 "func_for_data": func3,
        >>>                 "func_args": (1000,),
        >>>                 "interval" : 1000,
        >>>                 "title" : "other other scatter plot",
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

        self.fig.canvas.set_window_title(self.title)

    def display(self):
        """display information
        """
        print(f"""
        =====================================================
        Configuration Information
        =====================================================

        title   : {self.title}
        
        """)
