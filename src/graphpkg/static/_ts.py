'''
Timeseries Specific Plotting Module

created by : Nishant Baheti <nishantbaheti.it19@gmail.com>
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

_FORMATS = {
    'YEAR': 'Y',
    'QUARTER': 'Q',
    'MONTH': 'M',
    'DAY': 'D',
    'HOUR': 'H',
    'MINUTE': 'MIN'
}


def plot_boxed_timeseries(df: pd.DataFrame, ts_col: str, data_col: str, box: Optional[str] = 'MONTH', figsize: Optional[tuple] = None):
    """
    Plot timeseries data integrated with boxplot to see window based data variation.

    Args:
        df (pd.DataFrame): pandas dataframe.
        ts_col (str): timeseries column name.
        data_col (str): data column name.
        box (Optional[str], optional): time box. Defaults to 'MONTH'.
        figsize (Optional[tuple], optional): figure size. Defaults to None.

    Returns:
        Figure, Axes: Matplotlib figure and axes.

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import pandas as pd
        >>> from graphpkg.static import plot_boxed_timeseries
        >>> size = 1000
        >>> df = pd.DataFrame({
        >>>     "data": np.random.normal(size=(size,)) * 100,
        >>>     "timestamps": pd.date_range(start='1/1/2018', periods=size, freq='MIN')
        >>> })
        >>> fig, ax = plot_boxed_timeseries(df, data_col='data', ts_col='timestamps', box='hour', figsize=(10, 5))
        >>> plt.tight_layout()
        >>> plt.show()
    """
    assert box is not None, f"{box} boxtype cant be None."
    box = box.upper()
    assert box in _FORMATS, f"{box} boxtype is not configured."

    df[box] = df[ts_col].dt.to_period(_FORMATS.get(box)).apply(str)

    _df = df.groupby(box, as_index=False).agg(
        min_values=pd.NamedAgg(column=data_col, aggfunc="min"),
        max_values=pd.NamedAgg(column=data_col, aggfunc="max")
    )

    _fig, _ax = plt.subplots(1, 1, figsize=figsize or (10, 5))
    _ax.fill_between(_df[box], _df['min_values'], _df['max_values'], alpha=0.3)
    sns.boxplot(data=df, x=box, y=data_col, ax=_ax)
    _ax.tick_params(axis='x', labelrotation=90)
    _ax.grid(True)

    return _fig, _ax


if __name__ == "__main__":
    pass
    