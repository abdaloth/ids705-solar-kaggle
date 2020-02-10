import numpy as np
from utils import get_data, get_HOG, plot_roc, make_submission
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from tqdm import tqdm

X, y = get_data()
X = np.stack([get_HOG(img) for img in X ])

# %%

clf = SVC(C=10, probability=True, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
prediction_scores = np.empty(y.shape[0],dtype='object')

for train_idx, val_idx in tqdm(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = y[train_idx]
    
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict_proba(X_val)[:,1]

    # Save the predictions for this fold
    prediction_scores[val_idx] = y_pred


plot_roc(y, prediction_scores)

# %%
X_test, test_ids = get_data(test=True)
X_test = np.stack([get_HOG(img) for img in X_test])

clf = clf.fit(X, y)
test_predictions = clf.predict_proba(X_test)[:,1]
make_submission(test_ids, test_predictions, fname='submissions/svc_10_hog_16_2_fulltrain.csv')