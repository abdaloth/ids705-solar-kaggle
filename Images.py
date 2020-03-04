from utils import make_submission, get_data
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure
# %%
#Start by generating two raw images

X, y = get_data(data_dir_path='./data/data',as_gray=False)

pos_image = X[3]
neg_image = X[1]


#Hog Processing
fd, pos_hog_image = hog(pos_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
fd, neg_hog_image = hog(neg_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

#Canny Processing
pos_canny = cv2.Canny(pos_image,100,200)
neg_canny = cv2.Canny(neg_image,100,200)

#Create figure to hold images
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
ax1.imshow(pos_image)
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_title('Raw')
ax1.set_ylabel('Positive Image')

# Rescale histogram for better display
pos_hog_image_rescaled = exposure.rescale_intensity(pos_hog_image, in_range=(0, 10))
neg_hog_image_rescaled = exposure.rescale_intensity(neg_hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(pos_image)
ax2.imshow(pos_hog_image_rescaled, cmap=plt.cm.gray, alpha=.4)

ax2.set_title('HOG')

ax3.axis('off')
ax3.imshow(pos_image)
ax3.imshow(pos_canny, cmap= plt.cm.gray, alpha=.4)
ax3.set_title('Canny Edge Detection')


ax4.imshow(neg_image)
ax4.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_ylabel('Negative Image')


ax5.axis('off')
ax5.imshow(neg_image)
ax5.imshow(neg_hog_image_rescaled, cmap = plt.cm.gray, alpha=.4)

              
ax6.axis('off')
ax6.imshow(neg_image)
ax6.imshow(neg_canny, cmap = plt.cm.gray, alpha=.4)
plt.suptitle('Image Processing Techniques')

plt.show()
 