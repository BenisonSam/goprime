from __future__ import print_function

import itertools
import os
import random
import time

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import add

from goprime.common import create_batches


############################
# AlphaGo Zero style network

class ResNet(object):
    def __init__(self, input_N=256, filter_N=256, n_stages=19,
                 kernel_width=3, kernel_height=3,
                 inpkern_width=3, inpkern_height=3):
        self.model = None

        # number of filters and dimensions of the initial input kernel
        self.input_N = input_N
        self.inpkern_width = inpkern_width
        self.inpkern_height = inpkern_height

        # base number of filters and dimensions of the followup resnet kernels
        self.filter_N = filter_N
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.n_stages = n_stages

    def create(self, width, height, n_planes):
        bn_axis = 3
        inp = Input(shape=(width, height, n_planes))

        x = inp
        x = Conv2D(self.input_N, (self.inpkern_width, self.inpkern_height), padding='same', name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)

        for i in range(self.n_stages):
            x = self.identity_block(x, [self.filter_N, self.filter_N], stage=i + 1, block='a')

        self.model = Model(inp, x)

        return self.model

    def identity_block(self, input_tensor, filters, stage, block):
        """The identity_block is the block that has no conv layer at shortcut

        # Arguments
            input_tensor: input tensor
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        """
        nb_filter1, nb_filter2 = filters
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = input_tensor
        x = Conv2D(nb_filter1, (self.kernel_width, self.kernel_height),
                   padding='same', name=conv_name_base + 'a')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (self.kernel_width, self.kernel_height),
                   padding='same', name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)

        x = add([x, input_tensor])
        return x


class AGZeroModel:
    def __init__(self, N, batch_size=32, archive_fit_samples=64, use_tpu=None, log_path='logs/tensorboard'):
        self.N = N
        self.batch_size = batch_size

        self.model = None
        self.archive_fit_samples = archive_fit_samples
        self.position_archive = []

        self.tpu_grpc_url = use_tpu
        tpu_name_environ_key = 'TPU_NAME'

        # Check has server got TPU
        if use_tpu is not False and tpu_name_environ_key in os.environ:
            tpu_name = os.environ[tpu_name_environ_key].strip()
            if tpu_name != "":
                self.is_tpu = True
                self.tpu_grpc_url = TPUClusterResolver(tpu=[os.environ[tpu_name_environ_key]]).get_master()
        # TODO write an if condition to validate and resolve the TPU url provided

        self.__loss_functions = ['categorical_crossentropy', 'binary_crossentropy']

        self.model_name = time.strftime('GM{0}-%y%m%dT%H%M%S').format('%02d' % N)
        # print(self.model_name)

        log_path = os.path.join(log_path, self.model_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.callback = TensorBoard(log_path)

    def create(self):
        bn_axis = 3

        N = self.N
        position = Input((N, N, 6))
        resnet = ResNet(n_stages=N)
        resnet.create(N, N, 6)
        x = resnet.model(position)

        dist = Conv2D(2, (1, 1))(x)
        dist = BatchNormalization(axis=bn_axis)(dist)
        dist = Activation('relu')(dist)
        dist = Flatten()(dist)
        dist = Dense(N * N + 1, activation='softmax', name='distribution')(dist)

        res = Conv2D(1, (1, 1))(x)
        res = BatchNormalization(axis=bn_axis)(res)
        res = Activation('relu')(res)
        res = Flatten()(res)
        res = Dense(256, activation='relu')(res)
        res = Dense(1, activation='sigmoid', name='result')(res)

        self.model = Model(position, [dist, res])
        self.model.compile(Adam(lr=2e-2), self.__loss_functions)

        self.callback.set_model(self.model)

        # check if TPU available
        if self.tpu_grpc_url is not None:
            self.model = tf.contrib.tpu.keras_to_tpu_model(
                self.model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(self.tpu_grpc_url)))

        self.model.summary()

    def fit_game(self, X_positions, result):
        X_posres = []

        for pos, dist in X_positions:
            X_posres.append((pos, dist, result))
            result = -result

        if len(self.position_archive) >= self.archive_fit_samples:
            archive_samples = random.sample(self.position_archive, self.archive_fit_samples)
        else:
            # initial case
            archive_samples = self.position_archive

        self.position_archive.extend(X_posres)

        # I'm going to some lengths to avoid the potentially overloaded + operator
        X_fit_samples = list(itertools.chain(X_posres, archive_samples))

        self.__fit_model(X_fit_samples, self.batch_size)

    def retrain_position_archive(self, batch_size=None):
        self.__fit_model(self.position_archive, batch_size if batch_size else self.batch_size * 8)

    def reduce_position_archive(self, ratio=0.5):
        try:
            self.position_archive = random.sample(self.position_archive, int(len(self.position_archive) * ratio))
        except:
            pass

    def __fit_model(self, X_fit_samples, batch_size):
        batch_no = 1
        X, y_dist, y_res = [], [], []

        X_shuffled = random.sample(X_fit_samples, len(X_fit_samples))
        X_shuffled = create_batches(X_shuffled, batch_size)

        for batch in X_shuffled:
            for pos, dist, res in batch:
                X.append(pos)
                y_dist.append(dist)
                y_res.append(float(res) / 2 + 0.5)

            logs = self.model.train_on_batch(np.array(X), [np.array(y_dist), np.array(y_res)])

            self.write_log(self.__loss_functions, logs, batch_no)

            batch_no += 1
            X, y_dist, y_res = [], [], []

    def write_log(self, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()

            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name

            self.callback.writer.add_summary(summary, batch_no)
            self.callback.writer.flush()

    def predict(self, X_positions):
        dist, res = self.model.predict(X_positions)
        res = np.array([r[0] * 2 - 1 for r in res])
        return [dist, res]

    def save(self, snapshot_id, save_archive=False):
        self.model.save_weights('%s.weights.h5' % (snapshot_id,))
        if save_archive:
            joblib.dump(self.position_archive, '%s.archive.joblib' % (snapshot_id,), compress=5)

    def load(self, snapshot_id):
        self.model.load_weights('%s.weights.h5' % (snapshot_id,))

        pos_fname = '%s.archive.joblib' % (snapshot_id,)
        try:
            self.position_archive = joblib.load(pos_fname)
        except:
            print('Warning: Could not load position archive %s' % (pos_fname,))

    def unload_pos_archive(self):
        self.position_archive = []

    def load_pos_archive(self, archive_file):
        try:
            print('Attempting to load position archive %s' % (archive_file,))
            self.position_archive = joblib.load(archive_file)
            print('Successfully loaded position archive %s' % (archive_file,))
            return True
        except:
            import traceback
            traceback.print_exc()
            print('Warning: Could not load position archive %s' % (archive_file,))
            return False

    def load_averaged(self, weights, log=None):
        new_weights = []
        loaded_weights = []

        for weight in weights:
            self.model.load_weights(weight)
            loaded_weights.append(self.model.get_weights())
            print("Read weight: {0}".format(weight))
            if log is not None:
                log("Read weight: {0}".format(weight), self.model_name)

        if len(loaded_weights) > 0:
            for weights_list_tuple in zip(*loaded_weights):
                new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])

            self.model.set_weights(new_weights)
        else:
            print("No weights to load. Initializing the model with random weights!")
