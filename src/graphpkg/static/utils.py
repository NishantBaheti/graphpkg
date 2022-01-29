"""
plotting utility

author : Nishant Baheti<nishantbaheti.it19@gmail.com>
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def plot_distribution(x: np.ndarray, indicate_data: Union[list, np.ndarray] = None,
                      figsize: tuple = None, kde: bool = True) -> None:
    """
    Plot distribution with additional informations.

    distribution and box plot from matplotlib and seaborn.

    Args:
        x (np.ndarray): input 1D array.
        indicate_data (Union[list, np.ndarray]) : data points to observe/indicate in plot.
                                                    Defaults to None.
        figsize (tuple, optional): figure size from matplotlib. Defaults to None.
        kde (bool, optional): kde parameter from seaborn. Defaults to True.

    Raises:
        AssertionError : only 1d arrays are allowed for input.

    Examples:
        >>> x = np.random.normal(size=(200,))
        >>> plot_distribution(x, indicate_data=[0.6])
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
    plt.show()


def plot_classfication_boundary(func, data: np.ndarray = None, size: int = 4, n_plot_cols: int = 1,
                                figsize: tuple = (5, 5), bound_details: int = 50) -> None:
    """
    Plot classification model's decision boundary.

    Args:
        func (function): Prediction function of ML model that.
        data (np.ndarray, optional): source data. restricted to 2 features and 1 target,
                                        in total 3 columns. Defaults to None.
        size (int, optional): size of canvas. Defaults to 4.
        n_plot_cols (int, optional): number of columns for number of plots. Defaults to 1.
        figsize (tuple, optional): matplotlib figure size. Defaults to (5, 5).
        bound_details (int, optional): how detailed the boundary should be. Defaults to 50.

    Raises:
        ValueError: If the input data's shape is not (k,3), k=number of rows.

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=500, n_features=2, random_state=25,
        >>>                             n_informative=1, n_classes=2, n_clusters_per_class=1,
        >>>                             n_repeated=0, n_redundant=0)
        >>> model = LogisticRegression().fit(X, y)
        >>> plot_classfication_boundary(func=model.predict, data=np.hstack((X,y.reshape(-1,1))),bound_details=100)
    """
    if data is not None:
        if not (len(data.shape) == 2 and data.shape[1] == 3):
            raise ValueError(
                "Only shape (k,3) data is allowed. For flat plotting purposes")

    loc_points = np.linspace(-size, size, bound_details)
    all_points = np.array(np.meshgrid(loc_points, loc_points)).T.reshape(-1, 2)

    probs = func(all_points)
    probs = probs if len(probs.shape) >= 2 else probs.reshape(-1, 1)

    n_plots = probs.shape[1]
    n_plot_rows = int(np.ceil(n_plots / n_plot_cols))

    _, _ax = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, figsize=figsize)
    _ax = _ax if isinstance(_ax, np.ndarray) else np.array([_ax])
    grid = _ax if len(_ax.shape) == 1 else _ax.reshape(n_plot_rows*n_plot_cols,)

    plotted = 0
    for ax_ele in grid:  # type: ignore
        sns.scatterplot(x=all_points[..., 0], y=all_points[..., 1],
                        hue=probs[..., plotted], palette='viridis', ax=ax_ele)

        if data is not None:
            sns.scatterplot(x=data[..., -3], y=data[..., -2],
                            hue=data[..., -1], palette='dark', ax=ax_ele, legend=False)
        plotted += 1
        if plotted == n_plots:
            break

    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
    plt.tight_layout()
    plt.show()


def multi_distplots(df: pd.DataFrame, n_cols: int = 4, bins: int = 20, kde: bool = True,
                    class_col: str = None, legend: bool = True, legend_loc: str = 'best',
                    figsize: tuple = None, palette: str = 'dark', grid_flag: bool = True,
                    xticks_rotation: int = 45) -> None:
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
        xticks_rotation (int, optional): xticks rotation angle. Defaults to 45.

    Examples:
        >>> from sklearn.datasets import fetch_california_housing
        >>> dataset = fetch_california_housing()
        >>> df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        >>> df['target'] = dataset.target 
        >>> multi_distplots(df, n_cols=2)
    """
    columns = df.columns
    n_labels = len(columns)
    n_cols = min(n_cols, n_labels)
    n_rows = int(np.ceil(n_labels / n_cols))

    _, _ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize or (n_cols*3, n_rows*2))

    if isinstance(_ax, (np.ndarray)):
        _ax = _ax.reshape((n_rows * n_cols))
    else:
        _ax = np.array([_ax])
    for idx, name in enumerate(columns):
        sns.histplot(data=df, x=name, hue=class_col, bins=bins, label=name, ax=_ax[idx],
                     legend=legend, palette=palette, kde=kde)
        if str(df[name].dtype) == 'object':
            _ax[idx].set_xticklabels(_ax[idx].get_xticklabels(), rotation=xticks_rotation)
        _ax[idx].grid(grid_flag)

    if legend:
        plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # x = np.random.normal(size=(200,))

    # plot_distribution(x, indicate_data=[0.6])

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.datasets import make_classification

    # X, y = make_classification(n_samples=500, n_features=2, random_state=25,
    #                             n_informative=1, n_classes=2, n_clusters_per_class=1,
    #                             n_repeated=0, n_redundant=0)

    # model = LogisticRegression().fit(X, y)

    # plot_classfication_boundary(func=model.predict, \
    #     data=np.hstack((X,y.reshape(-1,1))),bound_details=100)

    from sklearn.datasets import fetch_california_housing

    dataset = fetch_california_housing()

    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target

    multi_distplots(df, n_cols=2)

    pass
