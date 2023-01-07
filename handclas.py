import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import psycopg as psql

import configparser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf


config = configparser.ConfigParser()
config.read('config.ini')
# %%


ADC_channels = ['P1_1', 'P1_2', 'P2_1', 'P2_2', 'P3_1', 'P3_2', 'P4_1', 'P4_2', 'P5_1', 'P5_2']
IMU_channels = ['Euler_x', 'Euler_y', 'Euler_z', 'Acc_x', 'Acc_y', 'Acc_z']

sign_types = ['static', 'dynamic']
sign_types_dict = {'a': sign_types[0],
                   'ą': sign_types[1],
                   'b': sign_types[0],
                   'c': sign_types[0],
                   'ć': sign_types[1],
                   'ch': sign_types[1],
                   'cz': sign_types[1],
                   'd': sign_types[1],
                   'e': sign_types[0],
                   'ę': sign_types[1],
                   'f': sign_types[1],
                   'g': sign_types[1],
                   'h': sign_types[1],
                   'i': sign_types[0],
                   'j': sign_types[1],
                   'k': sign_types[1],
                   'l': sign_types[0],
                   'ł': sign_types[1],
                   'm': sign_types[0],
                   'n': sign_types[0],
                   'ń': sign_types[1],
                   'o': sign_types[0],
                   'ó': sign_types[1],
                   'p': sign_types[0],
                   'r': sign_types[0],
                   'rz': sign_types[1],
                   's': sign_types[0],
                   'ś': sign_types[1],
                   'sz': sign_types[1],
                   't': sign_types[0],
                   'u': sign_types[0],
                   'w': sign_types[0],
                   'y': sign_types[0],
                   'z': sign_types[1],
                   'ź': sign_types[1],
                   'ż': sign_types[1]}


SAMPLE_SIZE = 75


# %%
# LOADING
df = pd.read_csv("gesty.csv")

num_rows = df.shape[0] // SAMPLE_SIZE
num_ts = num_rows*SAMPLE_SIZE


# %%
# DATA AND LABELS
X = df.iloc[:num_ts,1:17].values
y = df.iloc[:num_ts,17].values


# %%
# ENCODING
cat_encode = layers.StringLookup(output_mode='int')
cat_encode.adapt(y)
y = cat_encode(y)


# %%
# SCALING
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = tf.cast(X, dtype='float32')


# %%
# CONVERTING
X = np.reshape(X,(num_ts//SAMPLE_SIZE, SAMPLE_SIZE, X.shape[1]))
y = np.reshape(y,(num_ts//SAMPLE_SIZE, SAMPLE_SIZE,36))


# %%
# SPLITTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
num_classes = len(np.unique(y_train))


# %%
# MODEL CONSTANTS
callbacks = [
    ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

# %%
# MODEL DEFINITION
model = Sequential()
model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
#model.add(StringLookup(num_bins=3, mask_value=None, salt=None, output_mode="int", sparse=False))
model.add(layers.LSTM(units=500 ))
model.add(layers.Dense(num_classes, activation='sigmoid'))


# %%
# MODEL COMPILATION
model.compile(loss='categorical_crossentropy', optimizer='Ftrl', metrics=['categorical_accuracy'])


# %%
# TRAINING
history = model.fit(X_train, y_train, epochs=10, batch_size=32,verbose=2,validation_data=(X_test,y_test))


# %%
# ACCURACY AND LOSS PLOTS
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['categorical_accuracy'])
plt.title('Model loss and accuracy')
plt.ylabel('Loss/Accuracy')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy'], loc='upper left')
plt.show()


#%%
# CONFUSION MATRIX
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

confusion_matrix = confusion_matrix(y_test_class, y_pred_class)
plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0, 1, 2], ['class 0', 'class 1', 'class 2'])
plt.yticks([0, 1, 2], ['class 0', 'class 1', 'class 2'])
plt.show()


#%%
# ROC PLOT
fpr = {}
tpr = {}
roc_auc = {}

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label='ROC curve for class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for all classes')
plt.legend(loc="lower right")
plt.show()
