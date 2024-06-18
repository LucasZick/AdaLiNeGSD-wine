import os
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ucimlrepo import fetch_ucirepo
from itertools import combinations

from adaline_sgd import AdalineSGD
from plot_decision_regions import plot_decision_regions_multiple

#fetch wine dataset
wine = fetch_ucirepo(id=109)
X = wine.data.features 
y = wine.data.targets

#select wine class
wine_class = 1
y = y.iloc[0:100, 0].values
y = np.where(y == wine_class, 0, 1)

#get column names
column_names = X.columns.values

#create directory to save plots
os.makedirs('plots', exist_ok=True)

#generate every combination from 0 to 12
feature_pairs = list(combinations(range(12), 2))

#create pdf to save plots
pdf_filename = 'plots/_all_plots_.pdf'
pdf = PdfPages(pdf_filename)

for idx, (column_x, column_y) in enumerate(feature_pairs):
    #create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    column_x_name = column_names[column_x]
    column_y_name = column_names[column_y]

    #extract features
    X_pair = X.iloc[0:100, [column_x, column_y]].values

    #standardize features
    X_std = np.copy(X_pair)
    X_std[:, 0] = (X_pair[:, 0] - X_pair[:, 0].mean()) / X_pair[:, 0].std()
    X_std[:, 1] = (X_pair[:, 1] - X_pair[:, 1].mean()) / X_pair[:, 1].std()

    #run adaline
    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)

    #plot regions
    plot_decision_regions_multiple(X_std, y, classifier=ada_sgd, ax=ax)
    ax.set_title(f'{column_x_name} vs {column_y_name}')
    ax.set_xlabel(f'{column_x_name} [standardized]')
    ax.set_ylabel(f'{column_y_name} [standardized]')
    ax.legend(loc='upper left')

    #add to the pdf
    pdf.savefig(fig)

    #save individually
    plot_filename = f'plots/{column_x_name}_vs_{column_y_name}.png'
    plt.savefig(plot_filename)
    plt.close()

#close pdf
pdf.close()

#open pdf
webbrowser.open_new(pdf_filename)
