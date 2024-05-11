from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Layer, BatchNormalization

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import mse
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import math
from lifelines.utils import concordance_index
import numpy as np
import random

import os
import matplotlib.pyplot as plt
import pickle
import datetime
import pandas as pd
from Utils.utils import FunctionHelper


class SurvModel:
    def __init__(self, xy, f_step=None, f_loss=None, use_clb=False, a=1, b=1, sample_size=-1, fix=None):

        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = xy

        if sample_size == -1:
            sample_size = self.x_train.shape[0]

        self.inputShape = self.x_train.shape[1]
        self.x_train_sampled = self.x_train[:sample_size]
        self.y_train_sampled = self.y_train[:sample_size]

        self.f_step = f_step
        self.f_loss = f_loss
        surv_reg2_func = [1, 9, 12, 1, 5, 12, 7, 9, 7]
        self.surv_reg2_f = FunctionHelper(surv_reg2_func).f
        self.use_clb = use_clb
        self.a = a
        self.b = b

        self.fix = fix

        w1 = (self.f_loss is not None)
        w2 = (self.f_step is not None)
        assert w1 or w2, "No Loss function provided"

    def set_config(self, optimizer, batch_size, epochs, test_mode=False, callbacks=None, verbose=0):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_mode = test_mode
        self.callbacks = callbacks
        self.verbose = verbose


    def fit(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.cindex])
        if self.test_mode:
            self.history = self.model.fit(self.x_train_sampled, self.y_train_sampled,
                                          validation_data=(self.x_val, self.y_val),
                                          batch_size=self.batch_size,
                                          # validation_split = 0.2,
                                          callbacks=self.callbacks,
                                          epochs=self.epochs,
                                          verbose=self.verbose)
        else:
            self.history = self.model.fit(self.x_train_sampled, self.y_train_sampled,
                                          batch_size=self.batch_size,
                                          # validation_split = 0.2,
                                          callbacks=self.callbacks,
                                          epochs=self.epochs,
                                          verbose=self.verbose)

    def create_model(self):
        # tf.keras.backend.clear_session()
        seed_num = 1
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed_num)
        random.seed(seed_num)
        tf.random.set_seed(seed_num)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
        my_init = keras.initializers.glorot_uniform(seed=seed_num)

        self.model = keras.Sequential(
            [
                keras.Input(shape=self.inputShape),
                layers.Dense(32, kernel_initializer=my_init, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5, seed=seed_num),

                layers.Dense(32, kernel_initializer=my_init, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def evaluate(self, x, ye):
        return self.evaluate_cindex(x, ye)

    def evaluate_test(self):
        return self.evaluate_cindex(self.x_test, self.y_test)

    def evaluate_cindex(self, x, ye):
        y_pred = self.model.predict(x)
        return self.cindex(ye, y_pred).numpy()

    def evo_function(self, x):
        return self.f_step.s2(x)

    def evo_leftf(self, x):
        return self.f_loss.sl(x)

    def evo_rightf(self, x):
        return self.f_loss.sr(x)

    def loss(self, y_true, y_pred):
        loss_value = self.surv_loss_approximation_function(y_true, y_pred)
        return loss_value

    def surv_loss_approximation_function(self, y_true, y_pred):
        y = y_true[:, 0]
        e = y_true[:, 1]
        y_diff = y - y_pred[:, 0]

        if self.fix is 'r':
            err = (e * self.a * self.evo_leftf(y_diff)) + ((1 - e) * self.b * self.surv_reg2_f.sr(y_diff))

        elif self.fix == 'l':
            err = (e * self.a * self.surv_reg2_f.sl(y_diff)) + ((1 - e) * self.b * self.evo_rightf(y_diff))

        else:
            err = (e * self.a * self.evo_leftf(y_diff)) + ((1 - e) * self.b * self.evo_rightf(y_diff))

        return err


    @staticmethod
    def mse_surv(y_true, y_pred):
        y = y_true[:, 0]
        e = y_true[:, 1]
        y_diff = y-y_pred[:,0]
        return e * K.square(y_diff) + (1 - e) * K.relu(y_diff)


    @staticmethod
    def cindex(y_true, y_pred):
        y = y_true[:, 0]
        e = y_true[:, 1]
        ydiff = y[tf.newaxis, :] - y[:, tf.newaxis]
        yij = K.cast(K.greater(ydiff, 0), K.floatx()) + K.cast(K.equal(ydiff, 0), K.floatx()) * K.cast(
            e[:, tf.newaxis] != e[tf.newaxis, :], K.floatx())  # yi > yj
        is_valid_pair = yij * e[:, tf.newaxis]

        ypdiff = tf.transpose(y_pred) - y_pred
        ypij = K.cast(K.greater(ypdiff, 0), K.floatx()) + 0.5 * K.cast(K.equal(ypdiff, 0), K.floatx())  # yi > yj
        cidx = (K.sum(ypij * is_valid_pair)) / K.sum(is_valid_pair)
        return tf.cond(tf.math.is_nan(cidx), lambda: 0.0, lambda: cidx)



class FlchainModel(SurvModel):
    def create_model(self):
        # tf.keras.backend.clear_session()
        seed_num = 1
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed_num)
        random.seed(seed_num)
        tf.random.set_seed(seed_num)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
        my_init = keras.initializers.glorot_uniform(seed=seed_num)

        self.model = keras.Sequential(
            [
                keras.Input(shape=self.inputShape),
                layers.Dense(32, kernel_initializer=my_init, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5, seed=seed_num),

                layers.Dense(32, kernel_initializer=my_init, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )


class NwtcoModel(SurvModel):
    def create_model(self):
        # tf.keras.backend.clear_session()
        seed_num = 1
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed_num)
        random.seed(seed_num)
        tf.random.set_seed(seed_num)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
        my_init = keras.initializers.glorot_uniform(seed=seed_num)

        self.model = keras.Sequential(
            [
                keras.Input(shape=self.inputShape),
                layers.Dense(32, kernel_initializer=my_init, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5, seed=seed_num),

                layers.Dense(32, kernel_initializer=my_init, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )


class SupportModel(SurvModel):
    def create_model(self):
        # tf.keras.backend.clear_session()
        seed_num = 1
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed_num)
        random.seed(seed_num)
        tf.random.set_seed(seed_num)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
        my_init = keras.initializers.glorot_uniform(seed=seed_num)

        self.model = keras.Sequential(
            [
                keras.Input(shape=self.inputShape),
                layers.Dense(32, kernel_initializer=my_init, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5, seed=seed_num),

                layers.Dense(32, kernel_initializer=my_init, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )


class DebugCallback(keras.callbacks.Callback):
    # def on_train_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Starting training; got log keys: {}".format(keys))
    #
    # def on_train_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop training; got log keys: {}".format(keys))

    # def on_epoch_begin(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print()
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        # print('\n loss {}'.format(logs['val_cindex_approximation_function']))

    # def on_test_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start testing; got log keys: {}".format(keys))
    #
    # def on_test_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop testing; got log keys: {}".format(keys))
    #
    # def on_predict_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start predicting; got log keys: {}".format(keys))
    #
    # def on_predict_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop predicting; got log keys: {}".format(keys))
    #
    # def on_train_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_train_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_test_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_test_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_predict_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_predict_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
