"""
plotting utility

author : Nishant Baheti<nishantbaheti.it19@gmail.com>
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from typing import Union

def plot_distribution(x: np.ndarray, indicate_data: Union[list, np.ndarray] = None, figsize: tuple=None, kde: bool=True) -> None:
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
    x = np.array(x) if isinstance(x,(list, tuple)) else x
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
    
    _, ax = plt.subplots(2,1,figsize=figsize or (5,5))  

    ax[0].boxplot(x, vert=False)
    ax[0].axvline(x=min_value, color = 'blue', lw = 2, label='min')
    ax[0].axvline(x=mean_value, color = 'k', lw = 2, label='mean')
    ax[0].axvline(x=median_value, color = 'red', lw = 2, label='median')
    ax[0].axvline(x=max_value, color = 'gray', lw = 2, label='max')

    sns.histplot(x, kde=kde, label='distribution', ax=ax[1], element='step')
    ax[1].axvline(x=min_value, color = 'blue', lw = 2, label='min')
    ax[1].axvline(x=mean_value, color = 'k', lw = 2, label='mean')
    ax[1].axvline(x=median_value, color = 'red', lw = 2, label='median')
    ax[1].axvline(x=max_value, color = 'gray', lw = 2, label='max')

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


if __name__ == "__main__":
    x = np.random.normal(size=(200,))

    plot_distribution(x,indicate_data=[0.6])
