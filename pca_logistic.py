# %%
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from utils import make_submission, get_data
# %%


X, y = get_data(data_dir_path='./data/data') #Read in Data

pca = PCA.fit(X) # Applying PCA to our data


#Visualizing the principal components. 
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()


    
