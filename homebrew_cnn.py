# %%
import numpy as np

from keras import layers as nn
from keras import optimizers as opt
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_auc_score
from keras.utils.np_utils import to_categorical
from utils import make_submission, get_data

from sklearn.model_selection import StratifiedKFold

# %%

BATCH_SIZE = 128
EPOCHS = 50

conv_params = {'kernel_size': 3,
               'padding': 'same',
               'activation': 'relu'}


def get_model():

    cnn_input = nn.Input(shape=(101, 101, 3))

    x = nn.Conv2D(filters=32, **conv_params)(cnn_input)
    x = nn.BatchNormalization(axis=-1)(x)
    x = nn.Conv2D(filters=32, **conv_params)(x)
    x = nn.BatchNormalization(axis=-1)(x)
    x = nn.MaxPool2D()(x)
    x = nn.Dropout(.25)(x)

    x = nn.Conv2D(filters=64, **conv_params)(x)
    x = nn.BatchNormalization(axis=-1)(x)
    x = nn.MaxPool2D()(x)
    x = nn.Conv2D(filters=64, **conv_params)(x)
    x = nn.BatchNormalization(axis=-1)(x)
    x = nn.MaxPool2D()(x)
    x = nn.Dropout(.25)(x)

    x = nn.Conv2D(filters=128, **conv_params)(x)
    x = nn.BatchNormalization(axis=-1)(x)
    x = nn.MaxPool2D()(x)
    x = nn.Conv2D(filters=128, **conv_params)(x)
    x = nn.BatchNormalization(axis=-1)(x)
    x = nn.MaxPool2D()(x)
    x = nn.Dropout(.25)(x)

    x = nn.Flatten()(x)

    x = nn.Dense(512, activation='relu')(x)
    x = nn.BatchNormalization()(x)
    x = nn.Dropout(.5)(x)

    output = nn.Dense(2, activation='softmax')(x)

    model = Model(inputs=cnn_input, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt.Adam(lr=1e-3, decay=2e-5))
    return model


model = get_model()
model.summary()
# %%
X, y = get_data(as_gray=False)
X = X/255.

imagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)

imagen.fit(X)


batch_per_epoch = len(X)/BATCH_SIZE

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

prediction_scores = np.empty(y.shape[0], dtype='object')
models = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model = get_model()
    y_train = to_categorical(y_train)
    model.fit_generator(imagen.flow(X_train,
                                    y_train,
                                    batch_size=BATCH_SIZE),
                        steps_per_epoch=batch_per_epoch,
                        epochs=EPOCHS, verbose=0)
    prediction_scores[val_idx] = model.predict(X_val,
                                               batch_size=BATCH_SIZE)[:, 1]
    cur_auc = roc_auc_score(y_val, prediction_scores[val_idx])
    print(cur_auc)
    if cur_auc < 0.8:
        break
    models.append(model)

print(roc_auc_score(y, prediction_scores))

# %%
X_test, test_ids = get_data(test=True, as_gray=False)
X_test = X_test/255.

test_predictions = np.mean([m.predict(X_test,
                                      batch_size=BATCH_SIZE)[:, 1] for m in models], axis=0)

make_submission(test_ids, test_predictions,
                'submissions/homebrew_cnn_CV.csv')
# %%
[m.save(f'data/models/model_fold_{i}.h5') for i, m in enumerate(models)]
