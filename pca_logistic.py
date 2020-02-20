# %%
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from utils import make_submission, get_data, plot_roc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
# %%


X, y = get_data(data_dir_path='./data/data') #Read in Data

nsamples, nx, ny = X.shape #Reshape data to be passed to PCA. 
X = X.reshape(nsamples, nx*ny)
pca = PCA() # Applying PCA to our data
pca.fit(X)


#Visualizing the principal components. 
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()

#Fit model to data using selected number of principal components [1000].

pca_limited = PCA(n_components = 1000)
X_transform  = pca_limited.fit_transform(X) #Transforming data.

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
prediction_scores = np.empty(y.shape[0],dtype='object')

for train_idx, val_idx in tqdm(skf.split(X_transform, y)):
    X_train, X_val = X_transform[train_idx], X_transform[val_idx]
    y_train = y[train_idx]
    
    logistic = LogisticRegression().fit(X_train,y_train)
    y_pred = logistic.predict_proba(X_val)[:,1]

    # Save the predictions for this fold
    prediction_scores[val_idx] = y_pred


plot_roc(y, prediction_scores)
    
