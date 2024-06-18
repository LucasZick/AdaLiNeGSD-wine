from adaline_sgd import AdalineSGD
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from plot_decision_regions import plot_decision_regions

#insert and select the dataframe
wine = fetch_ucirepo(id=109)
X = wine.data.features 
y = wine.data.targets

#select wine class
wine_class = 1
y = y.iloc[0:100, 0].values
y = np.where(y == wine_class, 0, 1)

#specify the columns being analyzed
column_x = 2
column_y = 6
column_x_name = X.columns.values[column_x]
column_y_name = X.columns.values[column_y]

X = X.iloc[0:100, [column_x, column_y]].values

#standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

#run adaline
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title(f'{column_x_name} vs {column_y_name}')
plt.xlabel(f"{column_x_name} [standardized]")
plt.ylabel(f"{column_y_name} [standardized]")
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')

plt.show()