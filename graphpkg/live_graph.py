"""
=======================================================================================================
    
    Devloped By : Nishant Baheti
    
-------------------------------------------------------------------------------------------------------

    - Work in progress

=======================================================================================================
"""

import matplotlib.pyplot
import matplotlib.animation 

class LiveTrend:
    """Live Graph Module

    Args:
        func_for_data (callable): Function to return x and y data point
            Example:
                >>> def get_new_data():
                >>>     return 1,10

                >>> def get_new_data():
                >>>     return None,10

                >>> def get_new_data():
                >>>     return 1,10
        interval (int): Interval to refresh data
        xlabel (str, optional): Label for x-axis. Defaults to None.
        ylabel (str, optional): Label for y-axis. Defaults to None.
        label (str, optional): Label for plot line. Defaults to None.
        title (str, optional): Title of trend chart. Defaults to None.
        window (int, optional): data point window. Defaults to None.

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
        func_for_data: callable, 
        interval: int, 
        xlabel: str = None, 
        ylabel: str = None, 
        label: str = None, 
        title: str = None, 
        window: int = None
        ) -> None:
        """Constructor

        """
        self.func_for_data = func_for_data
        self.interval = interval
        self.xlabel = xlabel or "x-axis"
        self.ylabel = ylabel or "y-axis"
        self.label = label or "Current Data"
        self.title = title or "Live Trend"
        self.window = window or 50
        self.xs = []
        self.ys = []
        self.fig = matplotlib.pyplot.figure()
        self.fig.canvas.set_window_title(self.title)
        self.ax = self.fig.add_subplot(111)
        self.ani = None
        self.counter = 0


    def _animate(self,i:int)-> None:
        """Animation method to update chart

        Args:
            i (int): counter
        """
        self.counter = i
        x_data, y_data = self.func_for_data()
        x_data = x_data or self.counter

        self.xs.append(x_data)
        self.ys.append(y_data)
        
        self.xs = self.xs[-self.window:]
        self.ys = self.ys[-self.window:]
        
        self.ax.clear()

        self.ax.set(
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel
        )
        self.ax.plot(self.xs, self.ys, label=self.label)
        self.ax.grid(color='grey', linewidth=0.3, visible=True)
        self.ax.legend()
        

    def start(self) -> None:
        """initiate the trend chart
        """
        self.ani = matplotlib.animation.FuncAnimation(
            self.fig, 
            self._animate,
            interval=self.interval
        )

    def print_configs(self) -> None:
        """Print information

            TODO
        """
        print("""
            Configuration Information
            -----------------------------------------------------
        """)




