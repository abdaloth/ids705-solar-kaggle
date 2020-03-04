"""
traditional machine learning approach with HOG features and an SVM classifier
"""
# %%
import numpy as np
from utils import get_data, get_HOG, plot_roc, make_submission, plot_prediction_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from tqdm import tqdm

import matplotlib.pyplot as plt

# %% 
imgs, y = get_data()

# compute image features

# arguments for the get_HOG method
hog_params = {'cell_size':(16, 16), 'block_size':(2, 2)}

# stack the feature arrays into one big input matrix
X = np.stack([get_HOG(img, **hog_params) for img in imgs ])

# %%

# set up the classifier
clf = SVC(C=10, probability=True, random_state=42)

# 5 fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
prediction_scores = np.empty(y.shape[0],dtype='object')

for train_idx, val_idx in tqdm(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = y[train_idx]
    
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict_proba(X_val)[:,1]

    # Save the predictions for this fold
    prediction_scores[val_idx] = y_pred


plt.title('SVM 5-fold cross validation ROC AUC')
plot_roc(y, prediction_scores)
plt.savefig('report/figures/svm_roc.png', dpi=300)

plot_prediction_samples(imgs, y, prediction_scores, 'SVM Prediction Samples')
plt.savefig('report/figures/svm_confmat.png', dpi=300)
# %%

# load and preprocess test data then create submission
X_test, test_ids = get_data(test=True)
X_test = np.stack([get_HOG(img, **hog_params) for img in X_test])

clf = clf.fit(X, y)
test_predictions = clf.predict_proba(X_test)[:,1]
make_submission(test_ids, test_predictions, fname='submissions/svc_10_hog_16_4_fulltrain.csv')