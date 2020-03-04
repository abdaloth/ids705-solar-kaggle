"""
collection of helper functions
"""

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.feature import hog

import sklearn.metrics as metrics

import matplotlib.pyplot as plt


def get_HOG(img, cell_size=(16, 16), block_size=(2, 2)):
    # return Histogram of Oriented Gradients (HOG) features
    return hog(img, pixels_per_cell=cell_size,
               cells_per_block=block_size)


def get_data(data_dir_path='./data', test=False, as_gray=True):
    if(not test):
        train_ids, train_labels = np.loadtxt(f'{data_dir_path}/labels_training.csv',
                                             delimiter=',',
                                             skiprows=1,
                                             unpack=True,
                                             dtype=int)
        X = np.stack([imread(f'{data_dir_path}/training/{id}.tif', as_gray) for id in train_ids])
        return X, train_labels
    else:
        test_ids = np.loadtxt(f'{data_dir_path}/sample_submission.csv',
                              delimiter=',',
                              skiprows=1,
                              usecols=0,
                              dtype=int)
        X = np.stack([imread(f'{data_dir_path}/testing/{id}.tif', as_gray) for id in test_ids])
        return X, test_ids


def get_confmat_sample_idx(labels, prediction_scores):

    # turn probabilities to labels
    predicted_labels = (prediction_scores > .5).astype(int)

    # create an array mask for each cell in the
    # confusion matrix
    tp_mask = ((labels == predicted_labels) & (labels == 1))
    fp_mask = labels < predicted_labels

    tn_mask = ((labels == predicted_labels) & (labels == 0))
    fn_mask = labels > predicted_labels

    # sort sample indices by confidence score
    sorted_idx = np.argsort(prediction_scores)

    # return indices of most confident
    # prediction types
    *_, tp_idx = sorted_idx[tp_mask[sorted_idx]]
    *_, fp_idx = sorted_idx[fp_mask[sorted_idx]]

    tn_idx, *_ = sorted_idx[tn_mask[sorted_idx]]
    fn_idx, *_ = sorted_idx[fn_mask[sorted_idx]]

    return (tp_idx, fp_idx, fn_idx, tn_idx)


def plot_prediction_samples(imgs, labels, prediction_scores, title):

    indices = get_confmat_sample_idx(labels, prediction_scores)

    ax_args = {'xticklabels': [],
               'yticklabels': [],
               'xticks': [],
               'yticks': []}

    titles = ['True Positive',
              'False Positive',
              'False Negative',
              'True Negative']

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()

    for i in range(4):
        idx = indices[i]
        ax[i].imshow(imgs[idx])
        ax[i].set(title=f'{titles[i]}\nprediction = {prediction_scores[idx]:.3f}',
                        **ax_args)
        pass
    
    if(title):
        fig.suptitle(title)
        pass
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])



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
    submission = pd.DataFrame({'id': test_ids, 'score': test_predictions})
    submission.to_csv(fname, index=False)
    pass
