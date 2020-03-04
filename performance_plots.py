# %%

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import get_data, plot_prediction_samples

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import metrics
# %%
imgs, labels = get_data(as_gray=False)
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
_, super_idx = next(sss.split(imgs, labels))
X, _, y, y_super_test = train_test_split(imgs, labels,
                                         test_size=0.2,
                                         random_state=42,
                                         stratify=labels)

svm_preds = np.load('data/svm_train_preds.npy',
                    allow_pickle=True)
cnn_preds = np.load('data/cv_cnn_train_preds.npy',
                    allow_pickle=True)
transfer_preds = np.load('data/transfer_cnn_train_preds.npy',
                         allow_pickle=True)

bagged_cnn_preds = np.load('data/cv_cnn_super_preds.npy',
                           allow_pickle=True)
ensemble_preds = 3*bagged_cnn_preds+transfer_preds+svm_preds[super_idx]
ensemble_preds = ensemble_preds/5

model_preds = [svm_preds, transfer_preds,
               bagged_cnn_preds, ensemble_preds]

model_names = ['SVM',
               'Transfer Learning',
               'Bagged CNN',
               'Weighted Voting']


y_vals = [labels, y_super_test, y_super_test, y_super_test]

# %%

fig, ax = plt.subplots(1, 1)
y_true = labels
for name, y_pred, y_true in zip(model_names, model_preds, y_vals):

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.roc_auc_score(y_true, y_pred)
    legend_string = f'{name} ; AUC = {auc:0.3f}'
    alpha = .4
    if(name == 'Weighted Voting'):
        alpha = 1
    ax.plot(fpr, tpr, label=legend_string, alpha=alpha)

ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
ax.set(xlabel='False Positive Rate',
       ylabel='True Positive Rate')

ax.grid(True)
ax.axis('square')
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

ax.legend(loc='lower center',
          ncol=3,
          fancybox=True,
          shadow=True,
          bbox_to_anchor=(0.5, -0.5))

fig.tight_layout()
plt.title('Comparing Performance of Different Models (ROC Curve)')
plt.savefig('report/figures/all_roc.png', dpi=300, bbox_inches='tight')

# %%

plot_prediction_samples(imgs, labels,
                        svm_preds, 'SVM Prediction Samples')
plt.savefig('report/figures/svm_confmat.png', dpi=300)

plot_prediction_samples(imgs[super_idx], y_super_test,
                        ensemble_preds, 'Weighted Voting Prediction Samples')
plt.savefig('report/figures/weighted_voting_confmat.png', dpi=300)

plot_prediction_samples(imgs[super_idx], y_super_test,
                        transfer_preds, 'Transfer Learning Prediction Samples')
plt.savefig('report/figures/transfer_cnn_confmat.png', dpi=300)

plot_prediction_samples(imgs[super_idx], y_super_test,
                        bagged_cnn_preds, 'Bagged CNN Prediction Samples')
plt.savefig('report/figures/bagged_cnn_confmat.png', dpi=300)

# %%

fig, ax = plt.subplots(1, 1)
y_true = labels
for name, y_pred, y_true in zip(model_names, model_preds, y_vals):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
    alpha = .4
    if(name == 'Weighted Voting'):
        alpha = 1

    ax.step(recall, precision, label=name, alpha=alpha)

ax.set(ylabel='Precision',xlabel='Recall')
ax.axis('square')
ax.set_ylim(0.3,1.01)
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

ax.legend(loc='lower center',
          ncol=3,
          fancybox=True,
          shadow=True,
          bbox_to_anchor=(0.5, -0.5))
fig.tight_layout()
plt.title('Comparing Performance of Different Models (Precision-Recall Curve)')
plt.savefig('report/figures/all_pr.png', dpi=300, bbox_inches='tight')

# %%

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, constrained_layout=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (_, val_idx) in enumerate(skf.split(X, y)):
    y_pred, y_true = cnn_preds[val_idx], y[val_idx]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
    ax1.step(recall, precision, label= f'subset {i}', alpha=alpha)
    auc = metrics.roc_auc_score(y_true, y_pred)
    legend_string = f'subset {i} ; AUC = {auc:0.3f}'
    alpha = .4
    ax2.plot(fpr, tpr, ls='--', label=legend_string, alpha=alpha)


precision, recall, _ = metrics.precision_recall_curve(y_super_test, bagged_cnn_preds, pos_label=1)
ax1.step(recall, precision, label='Bagged CNN')
ax1.set(ylabel='Precision',xlabel='Recall')

ax2.plot([0, 1], [0, 1], '--', color='gray', label='Chance')

fpr, tpr, _ = metrics.roc_curve(y_super_test, bagged_cnn_preds, pos_label=1)
auc = metrics.roc_auc_score(y_super_test, bagged_cnn_preds)
ax2.plot(fpr, tpr, label=f'Bagged CNN ; AUC = {auc:0.3f}')
ax2.set(xlabel='False Positive Rate',
       ylabel='True Positive Rate')

ax1.grid(True)
ax1.axis('square')

ax2.grid(True)
ax2.axis('square')
# Shrink current axis's height by 10% on the bottom

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

box = ax.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
labels = [f'subset {i}' for i in range(5)]
labels.append('Bagged CNN')
fig.tight_layout()
fig.suptitle('Bagging Multiple CNN Models, Trained on Subsets of The Same Data')
fig.legend(loc='lower center',
          ncol=3,
          labels=labels,
          fancybox=True,
          shadow=True)


plt.savefig('report/figures/bagged_cnn_compare.png', dpi=300, bbox_inches='tight')