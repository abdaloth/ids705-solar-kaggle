"""
collection of helper functions
"""

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.feature import hog

import sklearn.metrics as metrics

import matplotlib.pyplot as plt


def load_as_HOG(filename, cell_size=(16, 16), block_size=(2, 2)):
    # read image in single channel (greyscale)
    img = imread(filename, as_gray=True)
    # return Histogram of Oriented Gradients (HOG) features
    return hog(img, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2))


def get_data(data_dir_path='./data', test=False):
    if(not test):
        train_ids, train_labels = np.loadtxt(f'{data_dir_path}/labels_training.csv',
                                             delimiter=',',
                                             skiprows=1,
                                             unpack=True,
                                             dtype=int)
        X = np.stack([load_as_HOG(f'{data_dir_path}/training/{id}.tif')
                      for id in train_ids])
        return X, train_labels
    else:
        test_ids = np.loadtxt(f'{data_dir_path}/sample_submission.csv',
                                delimiter=',',
                                skiprows=1,
                                usecols=0,
                                dtype=int)
        X = np.stack([load_as_HOG(f'{data_dir_path}/testing/{id}.tif')
                      for id in test_ids])
        return X, test_ids



def plot_roc(y_true, y_pred):
    # copied from Kyle Bradbury's sample script
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.roc_auc_score(y_true, y_pred)
    legend_string = 'AUC = {:0.3f}'.format(auc)

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.axis('square')
    plt.legend()
    plt.tight_layout()


def make_submission(test_ids, test_predictions, fname='submission.csv'):
    submission = pd.DataFrame({'id':test_ids, 'score': test_predictions})
    submission.to_csv(fname, index=False)
    pass

