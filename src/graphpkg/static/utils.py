"""
plotting utility

author : Nishant Baheti<nishantbaheti.it19@gmail.com>
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from typing import Union


def plot_distribution(x: np.ndarray, indicate_data: Union[list, np.ndarray] = None, figsize: tuple = None, kde: bool = True) -> None:
    """
    Plot distribution with additional informations.

    distribution and box plot from matplotlib and seaborn.

    Args:
        x (np.ndarray): input 1D array.
        indicate_data (Union[list, np.ndarray]) : data points to observe/indicate in plot. Defaults to None.
        figsize (tuple, optional): figure size from matplotlib. Defaults to None.
        kde (bool, optional): kde parameter from seaborn. Defaults to True.

    Raises:
        AssertionError : only 1d arrays are allowed for input.
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

    _, ax = plt.subplots(2, 1, figsize=figsize or (5, 5))

    ax[0].boxplot(x, vert=False)
    ax[0].axvline(x=min_value, color='blue', lw=2, label='min')
    ax[0].axvline(x=mean_value, color='k', lw=2, label='mean')
    ax[0].axvline(x=median_value, color='red', lw=2, label='median')
    ax[0].axvline(x=max_value, color='gray', lw=2, label='max')

    sns.histplot(x, kde=kde, label='distribution', ax=ax[1], element='step')
    ax[1].axvline(x=min_value, color='blue', lw=2, label='min')
    ax[1].axvline(x=mean_value, color='k', lw=2, label='mean')
    ax[1].axvline(x=median_value, color='red', lw=2, label='median')
    ax[1].axvline(x=max_value, color='gray', lw=2, label='max')

    ax[1].axvline(x=mean_value + std_value, color='gray', ls='--')
    ax[1].axvline(x=mean_value - std_value, color='gray', ls='--')
    ax[1].axvline(x=mean_value + (2 * std_value), color='gray', ls='--')
    ax[1].axvline(x=mean_value - (2 * std_value), color='gray', ls='--')

    if indicate_data is not None:
        for ind_data in indicate_data:
            ax[1].axvline(x=ind_data, color='k', alpha=0.4, lw=3)
            ax[1].axvline(x=ind_data, color='k', lw=1, label=f"indicating {ind_data}")

    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()


def plot_classfication_boundary(func, data: np.ndarray = None, size: int = 4, n_plot_cols: int = 1,
                                figsize: tuple = (5, 5), bound_details: int = 50) -> None:
    """
    Plot classification model's decision boundary.

    Args:
        func (function): Prediction function of ML model that.
        data (np.ndarray, optional): source data. restricted to 2 features and 1 target, in total 3 columns. 
                                        Defaults to None.
        size (int, optional): size of canvas. Defaults to 4.
        n_plot_cols (int, optional): number of columns for number of plots. Defaults to 1.
        figsize (tuple, optional): matplotlib figure size. Defaults to (5, 5).
        bound_details (int, optional): how detailed the boundary should be. Defaults to 50.

    Raises:
        ValueError: If the input data's shape is not (k,3), k=number of rows.
    """
    if data is not None:
        if not (len(data.shape) == 2 and data.shape[1] == 3):
            raise ValueError("Only shape (k,3) data is allowed. For flat plotting purposes")

    loc_points = np.linspace(-size, size, bound_details)
    all_points = np.array(np.meshgrid(loc_points, loc_points)).T.reshape(-1, 2)
    
    probs = func(all_points)
    probs = probs if len(probs.shape) >= 2 else probs.reshape(-1, 1)
    
    n_plots = probs.shape[1]
    n_plot_rows = int(np.ceil(n_plots / n_plot_cols))

    _, ax = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, figsize=figsize)
    ax = ax if isinstance(ax, np.ndarray) else np.array([ax])
    grid = ax if len(ax.shape) == 1 else ax.reshape(n_plot_rows*n_plot_cols,)
    
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



if __name__ == "__main__":
    # x = np.random.normal(size=(200,))

    # plot_distribution(x, indicate_data=[0.6])

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, n_features=2, random_state=25,
                                n_informative=1, n_classes=2, n_clusters_per_class=1,
                                n_repeated=0, n_redundant=0)

    model = LogisticRegression().fit(X, y)

    plot_classfication_boundary(func=model.predict, \
        data=np.hstack((X,y.reshape(-1,1))),bound_details=100)
