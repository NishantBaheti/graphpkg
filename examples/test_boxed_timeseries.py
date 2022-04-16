import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from graphpkg.static import plot_boxed_timeseries

size = 1000
df = pd.DataFrame({
    "data": np.random.normal(size=(size,)) * 100,
    "timestamps": pd.date_range(start='1/1/2018', periods=size, freq='MIN')

})

fig, ax = plot_boxed_timeseries(df, data_col='data', ts_col='timestamps', box='hour', figsize=(10, 5))

plt.tight_layout()
plt.show()
