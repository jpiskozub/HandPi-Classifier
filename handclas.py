# %%
# IMPORTING MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# import psycopg as psql

from ast import literal_eval

import configparser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import tensorflow as tf

from keras.regularizers import l2

from tensorflow.keras.utils import timeseries_dataset_from_array, normalize

config = configparser.ConfigParser()
config.read('config.ini')

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

# %%
# CONSTANTS AND DICTIONARIES
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

test_acc_list = []
train_acc_list = []

# %%
# LOADING DATA
df = pd.read_csv("G:\Git_repos\HandPi-ETL\gesty.csv", converters={"exam_id": literal_eval})
df = df[df['exam_id'] != 15]
df.fillna(method='backfill', inplace=True)
print(f'NaN containment:{df.isnull().any()}')

# %%
# MODEL TRAINING WITH LEAVE-ONE-OUT CROSS VALIDATION OVER EACH SUBJECT
for i in pd.unique(df['exam_id']):

    # TRUNCATING DATASETS
    test_dframe = df.where(df['exam_id'] == i).dropna()
    train_dframe = df.where(df['exam_id'] != i).dropna()

    train_num_rows = train_dframe.shape[0] // SAMPLE_SIZE
    train_num_ts = train_num_rows * SAMPLE_SIZE

    test_num_rows = test_dframe.shape[0] // SAMPLE_SIZE
    test_num_ts = test_num_rows * SAMPLE_SIZE

    X_train_dframe = train_dframe.iloc[:train_num_ts, 1:17].values
    X_test_dframe = test_dframe.iloc[:test_num_ts, 1:17].values
    Y_train_dframe = train_dframe.iloc[:train_num_ts, 17].values
    Y_test_dframe = test_dframe.iloc[:test_num_ts, 17].values
    num_classes = len(np.unique(df['sign']))

    # CONVERTING & ENCODING LABELS
    Y_train_resh = np.reshape(Y_train_dframe, (train_num_ts // SAMPLE_SIZE, SAMPLE_SIZE, 1))
    Y_test_resh = np.reshape(Y_test_dframe, (test_num_ts // SAMPLE_SIZE, SAMPLE_SIZE, 1))

    Y_train = pd.get_dummies(Y_train_resh[:, 1, :].flatten())
    Y_test = pd.get_dummies(Y_test_resh[:, 1, :].flatten())

    # CONVERTING & SCALING VALUES
    scaler = MinMaxScaler()
    X_train_scaled = tf.cast(scaler.fit_transform(X_train_dframe), dtype='float32')
    X_test_scaled = tf.cast(scaler.fit_transform(X_test_dframe), dtype='float32')

    X_train = np.reshape(X_train_scaled, (train_num_ts // SAMPLE_SIZE, SAMPLE_SIZE, X_train_scaled.shape[1]))
    X_test = np.reshape(X_test_scaled, (test_num_ts // SAMPLE_SIZE, SAMPLE_SIZE, X_test_scaled.shape[1]))

    # SAVING DATASETS
    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train_resh, Y_train_enc))
    # test_dataset = tf.data.Dataset.from_tensor_slices((X_test_resh, Y_test_enc))

    train_dataset = (X_train, Y_train)
    test_dataset = (X_test, Y_test)

    # MODEL CONSTANTS
    LAYERS = np.dot(2, [75, 75, 75])  # number of units in hidden and output layers
    M_TRAIN = X_train.shape[0]  # number of training examples (2D)
    M_TEST = X_test.shape[0]  # number of test examples (2D),full=X_test.shape[0]
    N = X_train.shape[2]  # number of features
    BATCH = M_TRAIN //10   # batch size
    EPOCH = 100  # number of epochs
    LR = 2e-3  # learning rate of the gradient descent
    LAMBD = 3e-2  # lambda in L2 regularizaion
    DP = 0.0  # dropout rate
    RDP = 0.0  # recurrent dropout rate

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # MODEL CALLBACKS
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint = ModelCheckpoint(
        filepath=log_dir + '/models/' + 'model.{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5',
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max')

    lr_decay = ReduceLROnPlateau(monitor='loss',
                                 patience=1, verbose=1,
                                 factor=0.5, min_lr=1e-8)

    early_stop = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0,
                               patience=5, verbose=1, mode='auto',
                               baseline=0, restore_best_weights=True)

    initial_learning_rate = LR
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    # MODEL DEFINITION
    model = Sequential()
    model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))

    model.add(layers.GRU(units=LAYERS[0],
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                         dropout=DP, recurrent_dropout=RDP,
                         return_sequences=True, return_state=False,
                         stateful=False, unroll=False))
    model.add(layers.BatchNormalization())
    model.add(layers.GRU(units=LAYERS[1],
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                         dropout=DP, recurrent_dropout=RDP,
                         return_sequences=True, return_state=False,
                         stateful=False, unroll=False))
    model.add(layers.BatchNormalization())
    model.add(layers.GRU(units=LAYERS[1],
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                         dropout=DP, recurrent_dropout=RDP,
                         return_sequences=True, return_state=False,
                         stateful=False, unroll=False))
    model.add(layers.BatchNormalization())
    model.add(layers.GRU(units=LAYERS[1],
                     activation='tanh', recurrent_activation='hard_sigmoid',
                     kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                     dropout=DP, recurrent_dropout=RDP,
                     return_sequences=True, return_state=False,
                     stateful=False, unroll=False))
    model.add(layers.BatchNormalization())
    model.add(layers.GRU(units=LAYERS[1],
                     activation='tanh', recurrent_activation='hard_sigmoid',
                     kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                     dropout=DP, recurrent_dropout=RDP,
                     return_sequences=True, return_state=False,
                     stateful=False, unroll=False))
    model.add(layers.BatchNormalization())
    model.add(layers.GRU(units=LAYERS[2],
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                         dropout=DP, recurrent_dropout=RDP,
                         return_sequences=False, return_state=False,
                         stateful=False, unroll=False))

    # model.add(layers.BatchNormalization())
    # model.add(layers.Dense(36, activation='softmax'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dense(36, activation='softmax'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dense(36, activation='softmax'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dense(36, activation='softmax'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(Y_test.shape[1], activation='softmax'))

    opt = optimizers.Adam(learning_rate=LR, clipnorm=1.)

    # MODEL COMPILATION
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

    print(model.summary())

    # TRAINING
    history = model.fit(X_train, Y_train, epochs=EPOCH, batch_size=BATCH,
                        shuffle=False, validation_data=test_dataset
                        , callbacks=[
             lr_decay
            , checkpoint
            #, early_stop
            , tensorboard_callback]
                        )

    train_loss, train_acc = model.evaluate(X_train, Y_train,
                                           batch_size=M_TRAIN, verbose=0)
    test_loss, test_acc = model.evaluate(X_test,  Y_test,
                                         batch_size=M_TEST, verbose=0)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


    # ACCURACY AND LOSS PLOTS
    print(max(history.history['val_categorical_accuracy']))

    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model loss and accuracy')
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.show()

    # CONFUSION MATRIX
    oh_dict = dict(zip([i for i in range(num_classes)], list(sign_types_dict.keys())))

    y_pred = model.predict(X_test)
    y_int_pred_class = np.argmax(y_pred, axis=1)
    y_int_test_class = np.argmax(Y_test.values, axis=1)

    y_test_class = [oh_dict[i] for i in y_int_test_class]
    y_pred_class = [oh_dict[i] for i in y_int_pred_class]

    confusion_mat = confusion_matrix(y_test_class, y_pred_class, labels=list(sign_types_dict.keys()))
    plt.imshow(confusion_mat, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([cl for cl in range(num_classes)], sign_types_dict.keys())
    plt.yticks([cl for cl in range(num_classes)], sign_types_dict.keys())
    plt.show()

    tf.keras.backend.clear_session()

# tf.math.confusion_matrix(Y_test,y_pred)
# #%%
# # ROC PLOT
# fpr = {}
# tpr = {}
# roc_auc = {}
#
# for i in range(3):
#     fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# plt.figure()
# for i in range(3):
#     plt.plot(fpr[i], tpr[i], label='ROC curve for class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curves for all classes')
# plt.legend(loc="lower right")
# plt.show()

# %%
