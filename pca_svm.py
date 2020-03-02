# %%
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from utils import make_submission, get_data, plot_roc, get_pca_data
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
# %%


X, y = get_data(data_dir_path='./data/data') #Read in Data
#Reshape data
nsamples, nx, ny = X.shape
X = X.reshape((nsamples,nx*ny))

#Standardize data before PCA 
X = normalize(X)
#Transform data with PCA
pca_limited = PCA(n_components = 1000)
X_transform  = pca_limited.fit_transform(X)


#Doing a grid search to identify best hyperparameters for SVM. 

# C_vals = np.logspace(-2, 10, 13)
# gammas = np.logspace(-9, 3, 13)
# grid= dict(gamma=gammas, C=C_vals)
# splits = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=grid, cv= splits)
# grid.fit(X_transform, y)

# #Print 'Best' combination of hyperparameters. 
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))


#Fitting model over multiple folds with selected hyperparameters. 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
prediction_scores = np.empty(y.shape[0],dtype='object')

for train_idx, val_idx in tqdm(skf.split(X_transform, y)):
    X_train, X_val = X_transform[train_idx], X_transform[val_idx]
    y_train = y[train_idx]
    
    svm = SVC(probability=True).fit(X_train,y_train)
    y_pred = svm.predict_proba(X_val)[:,1]

    # Save the predictions for this fold
    prediction_scores[val_idx] = y_pred


plot_roc(y, prediction_scores)