"""
Devloped By : Nishant Baheti

A lot of things need to be added here. Will surely do.
"""

from typing import Iterable, List, TypeVar
from abc import ABC,abstractmethod
import matplotlib.pyplot
import matplotlib.animation
import numpy as np
from scipy import stats
import logging 
from graphpkg import __version__

__author__ = "Nishant Baheti"
__copyright__ = "Nishant Baheti"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

T = TypeVar('T')
A = TypeVar('A',int,float,list)

class Graph(ABC):
    """Graph Meta Class

    Args:
        fig (matplotlib.pyplot.figure): Matplotlib figure
        fig_spec (tuple): figure specification
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        label (str): label
        title (str): Graph title
    """

    def __init__(
        self,
        fig: matplotlib.pyplot.figure,
        fig_spec: tuple, 
        xlabel: str, 
        ylabel: str , 
        label: str, 
        title: str):
        """Constructor
        """
        self.xlabel = str(xlabel)
        self.ylabel = str(ylabel)
        self.label = str(label)
        self.title = str(title)
        self.fig = fig or matplotlib.pyplot.figure()
        self.fig.canvas.manager.set_window_title(self.title)
        self.ax = self.fig.add_subplot(*fig_spec)
        self.fig.tight_layout()

    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def display(self):
        pass

class LiveTrend(Graph):
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
        fig (matplotlib.pyplot.figure, optional): .Matplotlib figure. Defaults to None.
        fig_spec (tuple, optional): [description]. Matplotlib figure specification. Defaults to (1,2,(1,2)).
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
        super().__init__(
            fig = fig,
            fig_spec = fig_spec, 
            xlabel = str(xlabel), 
            ylabel = str(ylabel), 
            label = str(label), 
            title = str(title))
        self.func_for_data = func_for_data
        self.func_args = func_args
        self.interval = int(interval)
        self.window = int(window)
        self._max_line_plots = 3
        self._xs = []
        self._ys = [[] for i in range(self._max_line_plots)]
        self.ani = None
        self.counter = 0

    @property
    def xs(self)-> List[A]:
        """x-axis data list

        Returns:
            List[A]: x-axis list
        """
        return self._xs

    @property
    def ys(self)->List[A]:
        """y-axis data list

        Returns:
            List[A]: y-axis list of lists
        """
        return self._ys

    def _plot_single_line(self) -> None:
        """Plot single line in trend chart
        """
        self.ax.clear()
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.plot(self._xs, self._ys[0], marker = "o", markersize= 0.75, linewidth = 0.6, label=self.label)
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
                self._xs, self._ys[p_i], marker = "o", markersize= 0.75,linewidth = 0.6, label=self.label+"-"+str(p_i+1))
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
        if len(self._xs) > 0:
            if self._xs[-1] == x_data:
                plot_chart = False

        if plot_chart:
            if isinstance(y_data, float) or isinstance(y_data, int) or isinstance(y_data, str):
                self._xs.append(x_data)
                self._ys[0].append(y_data)
                self._xs = self._xs[-self.window:]
                self._ys = self._ys[-self.window:]
                self._plot_single_line()

            elif isinstance(y_data,list):
                
                self._xs.append(x_data)
                self._xs = self._xs[-self.window:]

                num_of_plots = self._max_line_plots if len(
                    y_data) > self._max_line_plots else len(y_data)
                for i in range(num_of_plots):
                    self._ys[i].append(y_data[i])
                    self._ys[i] = self._ys[i][-self.window:]
                
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

class LiveScatter(Graph):
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
        fig (matplotlib.pyplot.figure, optional): .Matplotlib figure. Defaults to None.
        fig_spec (tuple, optional): [description]. Matplotlib figure specification. Defaults to (1,1,1).
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
        super().__init__(
            fig = fig,
            fig_spec = fig_spec, 
            xlabel = str(xlabel), 
            ylabel = str(ylabel), 
            label = str(label), 
            title = str(title))
        self.func_for_data = func_for_data
        self.func_args = func_args
        self.interval = int(interval)
        self.window = int(window)
        self._max_scatter_plots = 3
        self._xs = []
        self._ys = [[] for i in range(self._max_scatter_plots)]
        self.ani = None
        self.counter = 0

    @property
    def xs(self)-> List[A]:
        """x-axis data list

        Returns:
            List[A]: x-axis list
        """
        return self._xs

    @property
    def ys(self)->List[A]:
        """y-axis data list

        Returns:
            List[A]: y-axis list of lists
        """
        return self._ys

    def _plot_single_scatter(self) -> None:
        """Plot single scatter chart
        """
        self.ax.clear()
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.scatter(self._xs, self._ys[0], s=[10], alpha= 0.6, label=self.label)
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
                self._xs, self._ys[p_i], s=[10], alpha= 0.6, label=self.label+"-"+str(p_i+1))
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
                self._xs.append(x_data)
                self._ys[0].append(y_data)
                self._xs = self._xs[-self.window:]
                self._ys = self._ys[-self.window:]
                self._plot_single_scatter()

            elif isinstance(y_data,list):
                
                self._xs.append(x_data)
                self._xs = self._xs[-self.window:]

                num_of_plots = self._max_scatter_plots if len(
                    y_data) > self._max_scatter_plots else len(y_data)
                for i in range(num_of_plots):
                    self._ys[i].append(y_data[i])
                    self._ys[i] = self._ys[i][-self.window:]
                
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

class LiveDistribution(Graph):
    """Live Distribution Graph Module

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
        fig (matplotlib.pyplot.figure, optional): .Matplotlib figure. Defaults to None.
        fig_spec (tuple, optional): [description]. Matplotlib figure specification. Defaults to (1,1,1).
        interval (int): Interval to refresh data in milliseconds.
        xlabel (str, optional): Label for x-axis. Defaults to "x-axis".
        ylabel (str, optional): Label for y-axis. Defaults to "y-axis".
        label (str, optional): Label for plot line. Defaults to "Current Data".
        title (str, optional): Title of Scatter chart. Defaults to "Live Scatter".
        window (int, optional): Data point window. Defaults to 500.

    Examples:
        >>> def func1():
                return None, random.randrange(1,100)
        >>> g1 = LiveDistribution(func_for_data=func1,interval=1000,title="plot 1 for range 1-100")
        >>> g1.start()
        >>> plt.show()
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
            window: int = 2000) -> None:
        """[summary]
        """
        super().__init__(
            fig = fig,
            fig_spec = fig_spec, 
            xlabel = str(xlabel), 
            ylabel = str(ylabel), 
            label = str(label), 
            title = str(title))
        self.func_for_data = func_for_data
        self.func_args = func_args
        self.interval = int(interval)
        self.window = int(window)
        self._max_plots = 3
        self._xs = []
        self._ys = [[] for i in range(self._max_plots)]
        self.ani = None
        self.counter = 0

    @property
    def xs(self)-> List[A]:
        """x-axis data list

        Returns:
            List[A]: x-axis list
        """
        return self._xs

    @property
    def ys(self)->List[A]:
        """y-axis data list

        Returns:
            List[A]: y-axis list of lists
        """
        return self._ys
        
    def _plot_single(self) -> None:
        """Plot single distribution chart
        """
        self.ax.clear()
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        
        x = np.array(self._ys[0])
        if x.shape[0] > 1:
            kernel = stats.gaussian_kde(x)
            count, bins, ignored = self.ax.hist(
                x=x, alpha=0.5, bins=30, density=True)
            self.ax.plot(bins, kernel(bins), label=self.label)
            self.ax.grid(color='grey', linewidth=0.3, visible=True)
            self.ax.legend(loc="upper left")

    def _plot_multi(self, num_of_plots: int) -> None:
        """Plot multi distribution chart

        Args:
            num_of_plots (int): Number of plots
        """
        self.ax.clear()
        self.ax.set(title=self.title, xlabel=self.xlabel,
                    ylabel=self.ylabel)
        
        for p_i in range(num_of_plots):
            x = np.array(self._ys[p_i])
            if x.shape[0] > 1:
                kernel = stats.gaussian_kde(x)
                count, bins, ignored = self.ax.hist(
                    x=x, alpha=0.5, bins=30, density=True)
                self.ax.plot(bins, kernel(bins), label=self.label+"-"+str(p_i))

        self.ax.grid(color='grey', linewidth=0.3, visible=True)
        self.ax.legend(loc="upper left")

    def _animate(self, i: int) -> None:
        """Animation method to update chart

        Args:
            i (int): counter
        """
        self.counter = i
        if self.func_args is not None:
            x_data, y_data = self.func_for_data(*self.func_args)
        else:
            x_data, y_data = self.func_for_data()

        if y_data is not None:
            if isinstance(y_data, float) or isinstance(y_data, int):
                self._xs.append(x_data)
                self._ys[0].append(y_data)
                self._xs = self._xs[-self.window:]
                self._ys = self._ys[-self.window:]
                self._plot_single()

            elif isinstance(y_data, list):

                self._xs.append(x_data)
                self._xs = self._xs[-self.window:]

                num_of_plots = self._max_plots if len(
                    y_data) > self._max_plots else len(y_data)
                for i in range(num_of_plots):
                    self._ys[i].append(y_data[i])
                    self._ys[i] = self._ys[i][-self.window:]

                self._plot_multi(num_of_plots)

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
