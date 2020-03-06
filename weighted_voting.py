"""
model that generates predictions by averaging the predictions 
of other models ( weighted by performance)
"""

import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce

# load the predictions (duplicate to increase weight)
submissions = [pd.read_csv('submissions/homebrew_cnn_CV.csv'),
               pd.read_csv('submissions/homebrew_cnn_CV.csv'),
               pd.read_csv('submissions/homebrew_cnn_CV.csv'),
               pd.read_csv('submissions/first_transfer_cnn.csv'),
               pd.read_csv('submissions/svc_10_hog_16_2_fulltrain.csv')]

# merge the dataframes

merged = reduce(lambda l, r: pd.merge(l, r, on='id'), submissions)

# identify the columns with the prediction scores
cols = merged.columns[1:].tolist()
score_cols = [col for col in cols if col.startswith('score')]

# generate the voted prediction
merged['score'] = merged[score_cols].mean(axis=1)

# save in submission format
merged = merged[['id', 'score']]
merged.to_csv('submissions/svc_transfer_homebrew_cnn_weighted_merge.csv', index=False)
