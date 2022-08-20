"""
plotting utility

author : Nishant Baheti<nishantbaheti.it19@gmail.com>
"""

from typing import Callable, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def plot_distribution(x: np.ndarray, kde: Optional[bool] = True, indicate_data: Optional[Union[list, np.ndarray]] = None,
                      figsize: Optional[tuple] = None) -> None:
    """
    Plot distribution with additional informations.

    distribution and box plot from matplotlib and seaborn.  

    Args:
        x (np.ndarray): input 1D array.
        kde (Optional[bool], optional): kde parameter from seaborn. Defaults to True.
        indicate_data (Optional[Union[list, np.ndarray]], optional): data points to observe/indicate in plot.
                                                                        Defaults to None.
        figsize (Optional[tuple], optional): figure size from matplotlib. Defaults to None.

    Raises:
        AssertionError : only 1d arrays are allowed for input.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from graphpkg.static import plot_distribution
        >>> x = np.random.normal(size=(200,))
        >>> plot_distribution(x, indicate_data=[0.6])
        >>> plt.show()
    """    
    
    x = np.array(x) if isinstance(x, (list, tuple)) else x
    assert len(x.shape) == 1, "only 1d arrays are allowed."

    min_value = x.min()
    max_value = x.max()
    mean_value = x.mean()
    std_value = x.std()
    median_value = np.median(x)
    mode_value = stats.mode(x)

    txt_summary = f"""
    Min     :   {min_value} 
    Max     :   {max_value} 
    Median  :   {median_value} 
    Mode    :   {mode_value} 
    Mean    :   {mean_value} 
    Std dev :   {std_value}   
    """
    print(txt_summary)

    _, _ax = plt.subplots(2, 1, figsize=figsize or (5, 5))

    _ax[0].boxplot(x, vert=False)
    _ax[0].axvline(x=min_value, color='blue', lw=2, label='min')
    _ax[0].axvline(x=mean_value, color='k', lw=2, label='mean')
    _ax[0].axvline(x=median_value, color='red', lw=2, label='median')
    _ax[0].axvline(x=max_value, color='gray', lw=2, label='max')

    sns.histplot(x, kde=kde, label='distribution', ax=_ax[1], element='step')
    _ax[1].axvline(x=min_value, color='blue', lw=2, label='min')
    _ax[1].axvline(x=mean_value, color='k', lw=2, label='mean')
    _ax[1].axvline(x=median_value, color='red', lw=2, label='median')
    _ax[1].axvline(x=max_value, color='gray', lw=2, label='max')

    _ax[1].axvline(x=mean_value + std_value, color='gray', ls='--')
    _ax[1].axvline(x=mean_value - std_value, color='gray', ls='--')
    _ax[1].axvline(x=mean_value + (2 * std_value), color='gray', ls='--')
    _ax[1].axvline(x=mean_value - (2 * std_value), color='gray', ls='--')

    if indicate_data is not None:
        for ind_data in indicate_data:
            _ax[1].axvline(x=ind_data, color='k', alpha=0.4, lw=3)
            _ax[1].axvline(x=ind_data, color='k', lw=1, label=f"indicating {ind_data}")

    plt.tight_layout()
    plt.legend(loc='upper right')


def adjust_multiplots(n_plots: int, n_cols: int, figsize: Union[tuple, None]):
    """
    Adjust multiple plots in matplotplot subplots.

    Args:
        n_plots (int): Number of plots.
        n_cols (int): Number of columns.
        figsize (Union[tuple, None]): figsize.

    Returns:
        matplolib figure, matplotlib subplot axes.

    Examples:
        >>> fig, ax = adjust_multiplots(n_plots=9, n_cols=3, figsize=(15,15))
    """
    n_cols = min(n_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    _fig, _ax = plt.subplots(ncols=n_cols, nrows=n_rows,
                             figsize=figsize or (n_cols*3, n_rows*3))

    if isinstance(_ax, (np.ndarray)):
        _ax = _ax.reshape((n_rows * n_cols))
    else:
        _ax = np.array([_ax])
    return _fig, _ax


def create_mesh(size: int, pts_details: int) -> list:
    """
    Create a mesh grid.

    Args:
        size (int): size/params of canvas.
        pts_details (int): detailed number of points.

    Returns:
        list: list of mesh variables. Generally referred to xx and yy.

    Examples:
        >>> xx, yy = create_mesh(size=4, pts_details=100)
    """
    loc_points = np.linspace(-size, size, pts_details)
    return np.meshgrid(loc_points, loc_points)


def create_canvas(size: int, pts_details: int) -> np.ndarray:
    """
    Create a 2 Dimensional canvas.

    Args:
        size (int): size/params of canvas.
        pts_details (int): detailed number of points.

    Returns:
        np.ndarray: numpy array for canvas points.

    Examples:
        >>> all_points = create_canvas(size=4, pts_details=100)
    """
    all_points = np.array(create_mesh(size, pts_details)).T.reshape(-1, 2)
    return all_points


def plot_classification_boundary(func: Callable, data: np.ndarray = None, size: int = 4, n_plot_cols: int = 1,
                                figsize: tuple = (5, 5), canvas_details: int = 50, canvas_opacity: float = 0.5,
                                canvas_palette: str = 'coolwarm'):
    """
    Plot classification model's decision boundary.

    Args:
        func (function): Prediction function of ML model that.
        data (np.ndarray, optional): source data. restricted to 2 features and 1 target, \
            in total 3 columns. Defaults to None.
        size (int, optional): size of canvas. Defaults to 4.
        n_plot_cols (int, optional): number of columns for number of plots. Defaults to 1.
        figsize (tuple, optional): matplotlib figure size. Defaults to (5, 5).
        canvas_details (int, optional): how detailed the boundary should be. Defaults to 50.
        canvas_opacity (float, optional): Canvas transparency parameter. Defaults to 0.3.
        canvas_palette (str, optional): palette of canvas. Defaults to 'coolwarm'.

    Raises:
        ValueError: If the input data's shape is not (k,3), k=number of rows.

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> import matplotlib.pyplot as plt
        >>> X, y = make_classification(n_samples=500, n_features=2, random_state=25,
        >>>                             n_informative=1, n_classes=2, n_clusters_per_class=1,
        >>>                             n_repeated=0, n_redundant=0)
        >>> model = LogisticRegression().fit(X, y)
        >>> plot_classification_boundary(func=model.predict, data=np.hstack((X,y.reshape(-1,1))),canvas_details=100)
        >>> plt.show()
    """
    if data is not None:
        if not (len(data.shape) == 2 and data.shape[1] == 3):
            raise ValueError(
                "Only shape (k,3) data is allowed. For flat plotting purposes")

    xx, yy = create_mesh(size=size, pts_details=canvas_details)

    all_points = np.c_[xx.ravel(), yy.ravel()]

    probs = func(all_points)
    probs = probs if len(probs.shape) >= 2 else probs.reshape(-1, 1)

    n_plots = probs.shape[1]
    n_plot_rows = int(np.ceil(n_plots / n_plot_cols))

    fig, _ax = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, figsize=figsize)
    _ax = _ax if isinstance(_ax, np.ndarray) else np.array([_ax])
    grid = _ax if len(_ax.shape) == 1 else _ax.reshape(n_plot_rows*n_plot_cols,)

    plotted = 0
    for ax_ele in grid:  # type: ignore
        # sns.scatterplot(x=all_points[..., 0], y=all_points[..., 1],
        #                 hue=probs[..., plotted], palette=canvas_palette, ax=ax_ele,
        #                 alpha=canvas_opacity)
        zz = probs[:,plotted].reshape(xx.shape)
        ax_ele.contourf(xx, yy, zz, cmap=canvas_palette, alpha=canvas_opacity)

        if data is not None:
            sns.scatterplot(x=data[..., -3], y=data[..., -2],
                            hue=data[..., -1], palette='dark', ax=ax_ele, legend=False)
        plotted += 1
        if plotted == n_plots:
            break

    fig.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
    fig.tight_layout()
    return fig, _ax


def grid_classification_boundary(models_list: list, data: np.ndarray = None,
                                size: int = 4, n_plot_cols: int = 3, figsize: tuple = (5, 5),
                                canvas_details: int = 50, canvas_opacity: float = 0.4, canvas_palette='coolwarm') -> None:
    """
    Plot multiple plots of clasification boundaries for mulitple ml models.

    Only models are allowed with 1D prediction.

    Args:
        models_list (list): Models list of dictionary.
        data (np.ndarray, optional): source data. restricted to 2 features and 1 target,
                                        in total 3 columns. Defaults to None.
        size (int, optional): Size of canvas. Defaults to 4.
        n_plot_cols (int, optional): number of plot columns. Defaults to 3.
        figsize (tuple, optional): figure size. Defaults to (5, 5).
        canvas_details (int, optional): detailing in canvas. Defaults to 50.
        canvas_opacity (float, optional): Canvas transparency parameter. Defaults to 0.4.
        canvas_palette (str, optional): palette from matplotlib. Defaults to coolwarm.

    Raises:
        ValueError: Only 3 dimensional data, 2 features, 1 target is allowed.

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import make_classification
        >>> import matplotlib.pyplot as plt
        >>> X, y = make_classification(n_samples=500, n_features=2, random_state=25,
        >>>                             n_informative=1, n_classes=2, n_clusters_per_class=1,
        >>>                             n_repeated=0, n_redundant=0)
        >>> lr_model = LogisticRegression().fit(X, y)
        >>> dt_model = DecisionTreeClassifier().fit(X, y)
        >>> models_list = [{
        >>>     "name": "Logistic Regression Classifier",
        >>>     "function": lr_model.predict
        >>> },{
        >>>     "name": "Decision Tree Classifier",
        >>>     "function": dt_model.predict
        >>> }]
        >>> grid_classification_boundary(models_list=models_list, data=np.hstack((X, y.reshape(-1, 1))), 
        >>>                             figsize=(7,5), canvas_details=100)
        >>> plt.show()
    """

    if data is not None:
        if not (len(data.shape) == 2 and data.shape[1] == 3):
            raise ValueError(
                "Only shape (k,3) data is allowed. For flat plotting purposes")

    xx, yy = create_mesh(size=size, pts_details=canvas_details)
    all_points = np.c_[xx.ravel(), yy.ravel()]
    n_plots = len(models_list)

    _, _ax = adjust_multiplots(n_plots=n_plots, n_cols=n_plot_cols, figsize=figsize)

    for i_plot in range(n_plots):

        try:
            func = models_list[i_plot]["function"]
            probs = func(all_points)

            assert len(probs.shape) == 1
        except AssertionError:
            print(f"{i_plot} number's model's output is not 1D")
        finally:
            probs = probs if len(probs.shape) >= 2 else probs.reshape(-1, 1)
            # sns.scatterplot(x=all_points[..., 0], y=all_points[..., 1],
            #                 hue=probs[..., 0], palette=canvas_palette, ax=_ax[i_plot],
            #                 alpha=canvas_opacity)

            zz = probs[:, 0].reshape(xx.shape)
            _ax[i_plot].contourf(xx, yy, zz, cmap=canvas_palette, alpha=canvas_opacity)

            if data is not None:
                sns.scatterplot(x=data[..., -3], y=data[..., -2],
                                hue=data[..., -1], palette='dark', ax=_ax[i_plot], legend=False)

            _ax[i_plot].legend(bbox_to_anchor=(1.2, 1), loc='upper right')
            _ax[i_plot].set_title(models_list[i_plot]["name"])

    plt.tight_layout()


def multi_distplots(df: pd.DataFrame, n_cols: int = 4, bins: int = 20, kde: bool = True,
                    class_col: str = None, legend: bool = True, legend_loc: str = 'best',
                    figsize: tuple = None, palette: str = 'dark', grid_flag: bool = True,
                    xticks_rotation: int = 60) -> None:
    """
    Mulitple Distribution Plots using pandas dataframe.

    Seaborn's histplot is used for distribution with additional functionality to have multiple
    distributions in one grid.

    Args:
        df (pd.DataFrame): Input dataframe.
        n_cols (int, optional): Number of columns in the grid. Defaults to 4.
        bins (int, optional): number of bins in distribution. Defaults to 20.
        kde (bool, optional): kde estimation line & plot. Defaults to True.
        class_col (str, optional): class column name for distribution separation and legend.
                                Defaults to None.
        legend (bool, optional): put legend or not. Defaults to True.
        legend_loc (str, optional): where to put legend, takes inputs similar to matplotlib.pyplot.
                                    Defaults to 'best'.
        figsize (tuple, optional): figure size, similar to matplotlib.pyplot. Defaults to None.
        palette (str, optional): color palette, property from seaborn. Defaults to 'dark'.
        grid_flag (bool, optional): put grid or not. Defaults to True.
        xticks_rotation (int, optional): xticks rotation angle. Defaults to 60.

    Examples:
        >>> from sklearn.datasets import fetch_california_housing
        >>> import pandas as pd
        >>> import numpy as np
        >>> dataset = fetch_california_housing()
        >>> df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        >>> df['target'] = dataset.target
        >>> multi_distplots(df, n_cols=2)
        >>> plt.show()
    """
    columns = df.columns
    n_labels = len(columns)
    _, _ax = adjust_multiplots(n_plots=n_labels, n_cols=n_cols, figsize=figsize)

    for idx, name in enumerate(columns):
        sns.histplot(data=df, x=name, hue=class_col, bins=bins, label=name, ax=_ax[idx],
                     legend=legend, palette=palette, kde=kde)
        if str(df[name].dtype) == 'object':
            _ax[idx].tick_params(labelrotation=xticks_rotation)
        _ax[idx].grid(grid_flag)

    if legend:
        plt.legend(loc=legend_loc)
    plt.tight_layout()


if __name__ == "__main__":
    # x = np.random.normal(size=(200,))

    # plot_distribution(x, indicate_data=[0.6])
    # plt.show()

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.datasets import make_classification

    # X, y = make_classification(n_samples=500, n_features=2, random_state=25,
    #                             n_informative=2, n_classes=3, n_clusters_per_class=1,
    #                             n_repeated=0, n_redundant=0)

    # model = LogisticRegression().fit(X, y)

    # fig, ax = plot_classification_boundary(func=model.predict, \
    #     data=np.hstack((X,y.reshape(-1,1))),canvas_details=100)
    # plt.show()

    # fig, ax = plot_classification_boundary(func=model.predict_proba,
    #                                        data=np.hstack((X, y.reshape(-1, 1))), canvas_details=100)
    # plt.show()

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.datasets import make_classification

    # X, y = make_classification(n_samples=500, n_features=2, random_state=25,
    #                             n_informative=1, n_classes=2, n_clusters_per_class=1,
    #                             n_repeated=0, n_redundant=0)

    # lr_model = LogisticRegression().fit(X, y)
    # dt_model = DecisionTreeClassifier().fit(X, y)

    # models_list = [{
    #     "name": "Logistic Regression Classifier",
    #     "function": lr_model.predict
    # },{
    #     "name": "Decision Tree Classifier",
    #     "function": dt_model.predict
    # }]

    # grid_classification_boundary(models_list=models_list, data=np.hstack((X, y.reshape(-1, 1))),
    #                             figsize=(7,5), canvas_details=100)
    # plt.show()

    # from sklearn.datasets import fetch_california_housing

    # dataset = fetch_california_housing()

    # df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    # df['target'] = dataset.target

    # multi_distplots(df, n_cols=2)
    # plt.show()

    pass
