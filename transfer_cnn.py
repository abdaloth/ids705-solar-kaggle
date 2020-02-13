import numpy as np

from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras import layers as nn
from keras.callbacks import ModelCheckpoint, EarlyStopping

from skimage.io import imread
from sklearn.metrics import roc_auc_score

from utils import make_submission, get_data, ROCAUC




# %%
model = VGG16(include_top=False, input_shape=(101, 101, 3))
for layer in model.layers:
    layer.trainable = False

x = nn.GlobalAveragePooling2D()(model.output)
x = nn.Dense(1024, activation='relu')(x)
x = nn.Dropout(.5)(x)
x = nn.Dense(512, activation='relu')(x)
x = nn.Dropout(.3)(x)
output = nn.Dense(1, activation='sigmoid')(x)

model = Model(inputs=model.input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam')

model.summary()
# %%
X, y = get_data(as_gray=False)

batch_size = 128
ra = ROCAUC(batch_size)
es = EarlyStopping(monitor='val_auc', patience=2, mode='max')
mc = ModelCheckpoint(f'data/models/model.h5',
                     monitor='val_auc',
                     save_best_only=True,
                     mode='max', verbose=1)
model.fit(X, y,
          batch_size=batch_size,
          epochs=50,
          validation_split=.2,
          callbacks=[ra, es, mc])


# %%
model.fit(X, y, batch_size=batch_size, epochs=5)


# %%
X_test, test_ids = get_data(test=True, as_gray=False)

test_predictions = model.predict(X_test, batch_size=batch_size)
test_predictions = test_predictions.flatten()
make_submission(test_ids, test_predictions, 'submissions/first_transfer_cnn.csv')

