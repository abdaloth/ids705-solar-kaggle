# %%
from utils import make_submission, get_data, plot_roc, get_pca_data
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from tqdm import tqdm
from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras import layers as nn
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import make_submission, get_data, ROCAUC


# %%

#Canny edge detection attempts to detect lines that exist in images. Given our solar panel example, nested sets of lines could indicate the presence of solar panels on a roof.

X, y = get_data(data_dir_path='./data/data', as_gray = True)

#Multiply by 256 to get back to regular scale
X = X*256
 
#Convert x values to integers 
X = X.astype(np.uint8, copy=False)
edges = cv2.Canny(X[3],100,200)
plt.imshow(X[3])
plt.imshow(edges)
plt.show()

#Applying conversion algorithm to all the data 
for i in range (X.shape[0]):
    X[i] = cv2.Canny(X[i],100,200)

    
#Changing data more to fit proper shape

nsamples, nx, ny = X.shape
X = X.reshape((nsamples,nx*ny))

#Fitting model over multiple folds. 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
prediction_scores = np.empty(y.shape[0],dtype='object')

for train_idx, val_idx in tqdm(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = y[train_idx]
    
    svm = SVC(probability=True, gamma = 'scale').fit(X_train,y_train)
    y_pred = svm.predict_proba(X_val)[:,1]

    # Save the predictions for this fold
    prediction_scores[val_idx] = y_pred


plot_roc(y, prediction_scores)

#Read in test data and fit model. 

test_x, test_ids = get_data(data_dir_path='./data/data', as_gray = True. test=True)
test_x = test_x *256
test_x = test_x.astype(np.uint8, copy=False)
for i in range (test_x.shape[0]):
    test_x[i] = cv2.Canny(test_x[i],100,200)
    

    
nsamples, nx, ny = test_x.shape
test_x = test_x.reshape((nsamples,nx*ny))


clf = SVC(probability = True, gamma = 'scale').fit(X, y)
test_predictions = clf.predict_proba(test_x)[:,1]

make_submission(test_ids, test_predictions, fname='scv_edge_detection_fulltrain.csv')