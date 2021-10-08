import numpy as np
from graphpkg.static.utils import plot_distribution

x = np.random.normal(size=(200,))

plot_distribution(x)
