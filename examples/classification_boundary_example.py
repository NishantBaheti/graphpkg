

import numpy as np
from graphpkg.static import plot_classfication_boundary


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=2, random_state=25,
                            n_informative=1, n_classes=2, n_clusters_per_class=1,
                            n_repeated=0, n_redundant=0)
model = LogisticRegression().fit(X, y)
plot_classfication_boundary(func=model.predict, data=np.hstack((X,y.reshape(-1,1))),bound_details=100)