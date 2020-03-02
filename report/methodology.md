## Methodology

In the preceding sections, we contextualized the problem we were trying to solve, namely, identifying satellite images which contained solar panels.

From a technical standpoint, this constitutes a binary classification problem. The input to this problem are images which are labelled as belonging to either one of two classes:

* The class of images which _do_ have a solar panel in them;
* The class of images that _do not_ have a solar panel in them.

Methodologies to solve this problem distinguish themselves, primarily, by the way in which they transform the images into a set of meaningful features for classification and, moreover, by the manner by which they use these features to do the actual classification task.

We have tackled this problem with a four different methods:

1. Pre-processing data using a Histogram of Oriented Gradients (HOG) approach and, subsequently, classifying features using a Support Vector Machine (SVM) classifier;

2. Lorem Ipsum

3. Convolutional Neural Network with Transfer Learning

4. Lorem Ipsum

In all such models, our output was the estimated probability that an image belonged to the class of images that presented a solar panel (as opposed to a sheer binary prediction of the class itself).

In the following subsections, we present each of the afore cited approaches in greater detail.

### HOG-SVM

In our first attempt to classify the satellite images, pre-processed the data using a HOG approach. Then, we used an SVM classifier to classify images in our two classes of interest.

A HOG transform identifies pixels whose values are substantially different than their neighbouring pixels, thereby suggesting that they mark the edge of some object.

The intensity of most things change smoothly and continuously. When two different objects are juxtaposed, however, the intensity of their pixels may change abruptly in their borders. Hence, identifying large gradients in pixel intensity is an approach by which edges may be recognized. This, in turn, can be used to convey information about the existence of a solar panel in an image.

An SVM classifier, in its turn, separates classes by identifying the threshold which maximizes the distance between both classes to the threshold. One can think of identifying two parallel thresholds which separate both classes while being as far apart from each other as possible.

Both HOG and SVM are widely used techniques, both individually[^1](e.g. Google, 2011, Person Following using Histogram of Oriented Gradients. Available at https://patentimages.storage.googleapis.com/c6/2f/0e/e2c41049da1711/US20110026770A1.pdf) and combined[^2](e.g. Rajiv Kapoor, Rashmi Gupta, Le Hoang Son, Sudan Jha, Raghvendra Kumar,
Detection of Power Quality Event using Histogram of Oriented Gradients and Support Vector Machine, Measurement, Volume 120, 2018, Pages 52-75, ISSN 0263-2241, https://doi.org/10.1016/j.measurement.2018.02.008; Dadi, H. S., & Pillutla, G. K. M. (2016). Improved face recognition rate using HOG features and SVM classifier. IOSR Journal of Electronics and Communication Engineering, 11(4), 34-44.). For these reasons, we chose to use them.

In more technical terms, we performed a HOG transformation using cells of sizes 16 by 16 and blocks of sizes 2 by 2. We used the `get_hog` function from package `utils`.

For the classification part, we used the `SVC` class from `scikit-learn` , with a regularization parameter of 10. Our output, as in all models we developed, was the predicted probability that the image pertained to the class of images that contained a solar panel.

We analysed the performance of our model using 5-fold stratified cross-validation and analysing the ROC curve.

### (Method 2)

Lorem Ipsum

### Transfer Learning

Convolutional neural networks are a mainstream technique in image classification. Its popularity stems not only from its remarkable accuracy, but also from its ability to create its own features, so to speak. In the words of Zagoruyko and Komodakis (2015, p.4353) [^](Zagoruyko, S., & Komodakis, N. (2015). Learning to compare image patches via convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4353-4361).), CNNs can

>"learn directly from image
data (i.e., without resorting to manually-designed features)
a general similarity function for comparing image patches,
which is a task of fundamental importance for many computer vision problems"

We thus do not need to pre-process the image using HOG or other similar technique in order to identify edges, corners or other features for our analysis [^](To be fair, Simonyan, K., & Zisserman, A. [2014] subtract the mean RGB from each pixel value, so there is _some_ pre-processing, but much less than other non-neural network based techniques).

Moreover, rather than flattening out an image into an array of pixels -- thereby loosing information on which pixels neighbour each other, -- CNNs retain information on each pixel's location. To see the relevance of this fact, it suffices to note that shuffling an images' pixels changes it completely and, indeed, it has recently been shown that changing a single pixel can be enough to fool a neural network in some cases[^](see Su, J., Vargas, D. V., & Sakurai, K. (2019). One pixel attack for fooling deep neural networks. IEEE Transactions on Evolutionary Computation, 23(5), 828-841.)

CNNs consider each image in their entirety and, therefore, have been shown to perform better than many other approaches in image classification and computer vision.

The kind of CNN we use in our approach is a VGG16 [^](Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.).

As the name suggests, this is a 16 layer CNN. The many deep layers of this neural network, allied with the small 3 x 3 [^](We say small, but we could have said minimal, since a 3 x 3 filter is the smallest dimension possible to capture the notions of up/down, left/right) receptive field it uses, made the VGG16 far more efficient in classifying images than pre-existing neural networks.

The activation function of these 16 layers is a Rectified Linear Unit (ReLU) function, except on the output layer, where we use a sigmoid. For the last two layers before the output layer, we use dense layers with dropout rates of 0.5 and 0.3 respectively. We use dropout to help our neural network prevent overfitting.

We set our neural network to minimize a cross-entropy loss function using the Adam optimization algorithm [^](Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.). We use a batch size of 128 and train our model for 50 epochs.





Unfortunately, however, training a VGG16 CNN takes a very long time. We thus decided to employ it under a transfer learning approach.

Transfer learning occurs when





It takes as input an RGB image of size $101 \times 101 \times 3$ RGB image and passes it through the convolution layers mentioned above. In each layer,



### (Method 4)
