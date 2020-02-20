# %%
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from utils import make_submission, get_data, plot_roc, get_pca_data
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
# %%


X, y = get_data(data_dir_path='./data/data') #Read in Data

#Fit model to data using selected number of principal components [1000].

X_transform = get_pca_data(X, 1000)

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