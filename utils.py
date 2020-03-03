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


def get_misclassified_indices(labels, prediction_scores):

    # turn probabilities to labels
    predicted_labels = (prediction_scores > .5).astype(int)

    # create an array mask for misclassification
    missed_mask = labels != predicted_labels

    # sort sample indices by confidence score (desc)
    sorted_idx = np.argsort(prediction_scores)[::-1]

    # return sorted list of misclassified indices
    filtered_idx = sorted_idx[missed_mask[sorted_idx]]

    return filtered_idx


def plot_misclassifications(imgs, labels, prediction_scores, n=3):

    idx = get_misclassified_indices(labels, prediction_scores)
    fig, ax = plt.subplots(2, n)

    for i in range(n):
        high_conf_idx = idx[i]
        low_conf_idx = idx[-(i+1)]

        ax[0, i].imshow(imgs[high_conf_idx])
        ax[0, i].set(xticklabels=[],
                     yticklabels=[],
                     xticks=[],
                     yticks=[],
                     title=(f'label = {labels[high_conf_idx]}',
                            f'\nprediction = {prediction_scores[high_conf_idx]:.3f}'))

        ax[1, i].imshow(imgs[low_conf_idx])
        ax[1, i].set(xticklabels=[],
                     yticklabels=[],
                     xticks=[],
                     yticks=[],
                     title=(f'label = {labels[low_conf_idx]}',
                            f'\n prediction = {prediction_scores[low_conf_idx]:.3f}'))
        pass

    fig.tight_layout()


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

