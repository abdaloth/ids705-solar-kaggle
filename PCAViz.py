# %%
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from utils import make_submission, get_data, plot_roc, get_pca_data
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np
#%%

#Import data and get 2d PCA.
X, y = get_data(data_dir_path='./data/data')

two_comp_data = get_pca_data(X, 2)

#Visualize 2d PCA to determine if we can separate the classes with a small amount of data. 
component_1 = two_comp_data[:,0]
component_2 = two_comp_data[:,1]
colors= ['red' if l == 0 else 'blue' for l in y]
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA Analysis and Class Distribution')
plt.scatter(component_1, component_2, color = colors)
plt.show()