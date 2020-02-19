"""
collection of helper functions
"""

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.feature import hog

import sklearn.metrics as metrics

import matplotlib.pyplot as plt


from keras.callbacks import Callback
    
    
class ROCAUC(Callback):
    def __init__(self, BATCH_SIZE=128):
        super(ROCAUC, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE

    def on_train_begin(self, logs={}):
        if not ('val_auc' in self.params['metrics']):
            self.params['metrics'].append('val_auc')

    def on_epoch_end(self, epoch, logs={}):
        logs['val_auc'] = float('-inf')
        if(self.validation_data):
            logs['val_auc'] = metrics.roc_auc_score(self.validation_data[1],
                                            self.model.predict(self.validation_data[0],
                                                               batch_size=self.BATCH_SIZE))



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

