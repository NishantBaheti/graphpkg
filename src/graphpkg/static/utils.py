import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def plot_distribution(x: np.ndarray,figsize: tuple=None, kde: bool=True, hist: bool=True, rug: bool=True):
            
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

    sns.distplot(x,kde=kde,hist=hist,ax=ax[1],rug=rug, label='distribution')
    ax[1].axvline(x=min_value, color = 'blue', lw = 2, label='min')
    ax[1].axvline(x=mean_value, color = 'k', lw = 2, label='mean')
    ax[1].axvline(x=median_value, color = 'red', lw = 2, label='median')
    ax[1].axvline(x=max_value, color = 'gray', lw = 2, label='max')

    ax[1].axvline(x=mean_value + std_value, color='gray', ls='--')
    ax[1].axvline(x=mean_value - std_value, color='gray', ls='--')
    ax[1].axvline(x=mean_value + (2 * std_value), color='gray', ls='--')
    ax[1].axvline(x=mean_value - (2 * std_value), color='gray', ls='--')


    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    x = np.random.normal(size=(200,))

    plot_distribution(x)
