# %%
import numpy as np
from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras import layers as nn
from keras.callbacks import EarlyStopping

from utils import make_submission, get_data, plot_roc, plot_prediction_samples

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


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
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=y)

batch_size = 128
es = EarlyStopping(monitor='val_loss', patience=2, mode='min')

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(X_val, y_val),
          callbacks=[es])


# %%
prediction_scores = model.predict(X_val, batch_size=128)
np.save('data/transfer_cnn_train_preds.npy',
        prediction_scores.flatten())
np.save('data/transfer_cnn_train_lbls.npy',
        y_val)
# %%
model.fit(X, y, batch_size=batch_size, epochs=5)


# %%
X_test, test_ids = get_data(test=True, as_gray=False)

test_predictions = model.predict(X_test, batch_size=batch_size)
test_predictions = test_predictions.flatten()
make_submission(test_ids, test_predictions,
                'submissions/first_transfer_cnn.csv')
