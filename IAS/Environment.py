# coding: utf8
import copy
import math
import time

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Ftrl

tensorflow.keras.backend.set_image_data_format("channels_last")


class Data_env(object):
    """
    The environment of Autodata
    """

    def __init__(self, data_pool, data_train, data_test, pool_component_num, batch_size, max_try_num=100, random_state=0):

        self.data_pool = data_pool
        self.data_train = data_train
        self.data_test = data_test

        self.data_train_history_list = []  # Track the history of training dataset

        self.batch_size = batch_size  # The number of data points added in the train dataset
        self.random_state = random_state
        self.pool_component_num = pool_component_num
        self.max_try_num = max_try_num

        self.feature_num = None

        self.try_num = 0

        self.pool_meta_info = {}

        self.current_model = None

        # Define FTRL-Proximal-like parameters
        self.optimizer = None
        self.alpha = 0.01  # Learning rate
        self.l1 = 0.1  # L1 regularization strength
        self.l2 = 0.1  # L2 regularization strength

        self.z_acc = 0
        self.n_acc = 0

        self.cur_score = None
        self.prev_score = None

        self.original_score = None

        self.action_space = [i for i in range(self.pool_component_num)]
        self.action_valid = []

        self.train_time = 0

        self.init_env()

    def init_env(self):
        # 1. Init some variables
        self.data_train_history_list = []
        self.try_num = 0

        # 2. Obtain the meta-info of the data in the pool.
        # Add selected label
        self.get_pool_meta_info()
        self.data_pool['Selected'] = [0 for _ in range(len(self.data_pool))]

        # 3. Init the model on the training dataset and test the model on the test dataset
        X_train, Y_train = self.get_training_dataset(self.data_train)
        self.current_model = self.model_initialize(X_train, Y_train)
        print('-' * 20 + "Init:" + '-' * 20)

        Train_score = self.model_test_score(X_train, Y_train)
        print(f"Train Score {Train_score}")

        X_test, Y_test = self.get_test_dataset()
        Test_score = self.model_test_score(X_test, Y_test)
        print(f"Test Score {Test_score}")

        self.cur_score = Test_score
        self.original_score = Test_score

        # 4. Check the valid action
        self.action_valid = []
        for i in range(self.pool_component_num):
            if len(self.data_pool.loc[
                       (self.data_pool['cluster_id'] == i) & (self.data_pool['Selected'] == 0)]) >= self.batch_size:
                self.action_valid.append(i)


    def step(self, action):
        """
        Execute the action
        """
        print(f"Action:{action}")
        # 1. Select data from the pool
        if len(
                self.data_pool.loc[(self.data_pool['cluster_id'] == action) & (
                        self.data_pool['Selected'] == 0)]) >= self.batch_size:
            m = self.batch_size
        else:
            m = len(
                self.data_pool.loc[
                    (self.data_pool['cluster_id'] == action) & (self.data_pool['Selected'] == 0)])

        if m == 0:
            done = False
            reward = 0
            return reward, done

        new_data_subset = self.data_pool.loc[
            (self.data_pool['cluster_id'] == action) & (self.data_pool['Selected'] == 0)].sample(
            n=m, replace=False, random_state=self.random_state).drop(['Selected'], axis=1)
        new_data_subset_index = new_data_subset.index.to_list()
        self.data_pool.loc[new_data_subset_index, 'Selected'] = 1
        new_data_list = [new_data_subset]

        # 2. Update the model on current training dataset
        X_train, Y_train = self.get_training_dataset(pd.concat(new_data_list).reset_index(drop=True).drop(['cluster_id'], axis=1))
        z = copy.deepcopy(self.z_acc)
        n = copy.deepcopy(self.n_acc)
        variables = copy.deepcopy(self.current_model.layers[-1].trainable_variables)
        time_start = time.time()
        self.current_model = self.model_training(X_train, Y_train)
        time_end = time.time()
        self.train_time += time_end-time_start

        # 4. Test the performance of the model

        X_test, Y_test = self.get_test_dataset()
        Test_score = self.model_test_score(X_test, Y_test)
        print(f"Test Score {Test_score}")


        # 5. Update the state, the reward and valid action
        reward = self.cur_score - Test_score
        if reward > 0:
            self.data_train_history_list.append(self.data_train)
            new_data_list.append(self.data_train)
            self.data_train = pd.concat(new_data_list).reset_index(drop=True)
            self.prev_score = self.cur_score
            self.cur_score = Test_score
        else:
            self.z_acc = copy.deepcopy(z)
            self.n_acc = copy.deepcopy(n)
            for i in range(len(self.n_acc)):
                self.current_model.layers[-1].trainable_variables[i].assign(variables[i])
        self.try_num += 1

        if self.try_num > self.max_try_num:
            print("Try too much time!!")
            done = True
            return reward, done
        else:
            done = False
            return reward, done


    # Some utils of the class
    def get_training_dataset(self, new_data_list):
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(new_data_list)
        n_past = 10
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:(dataset.shape[1] - 1)])
            dataY.append(dataset[i, dataset.shape[1] - 1])
        X_train = np.array(dataX)
        Y_train = np.array(dataY)
        return X_train, Y_train

    def get_test_dataset(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(self.data_test)
        n_past = 10
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:(dataset.shape[1] - 1)])
            dataY.append(dataset[i, dataset.shape[1] - 1])
        X_test = np.array(dataX)
        Y_test = np.array(dataY)

        return X_test, Y_test

    def model_initialize(self, X_train, Y_train):
        colnum = X_train.shape[2]
        n_past = 10

        # Define your LSTM model
        model = Sequential()
        model.add(LSTM(64, input_shape=(n_past, colnum)))
        model.add(Dense(1))

        # Define optimizer and initialize accumulators
        self.optimizer = Ftrl(learning_rate=0.01, l1_regularization_strength=0.1, l2_regularization_strength=0.1)
        self.z_acc = [tf.Variable(tf.zeros_like(var), trainable=False) for var in model.layers[-1].trainable_variables]
        self.n_acc = [tf.Variable(tf.zeros_like(var), trainable=False) for var in model.layers[-1].trainable_variables]

        # Training loop
        epochs = 8
        batch_size = 20
        steps_per_epoch = len(X_train) // batch_size

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = (step + 1) * batch_size
                X_batch = X_train[start:end]
                y_batch = Y_train[start:end]

                with tf.GradientTape() as tape:
                    predictions = model(X_batch)
                    loss = tf.keras.losses.mean_squared_error(y_batch, predictions)

                # Compute gradients
                gradients = tape.gradient(loss, model.layers[-1].trainable_variables)  # Exclude the last layer

                # Update the accumulators and apply the updated gradients
                for i, grad in enumerate(gradients):
                    z = self.z_acc[i]
                    n = self.n_acc[i]
                    alpha = self.optimizer.learning_rate
                    sigma = (tf.sqrt(n + tf.square(grad)) - tf.sqrt(n)) / self.alpha
                    # z += grad - (n * alpha * model.layers[-1].trainable_variables[i])
                    z.assign_add(grad - (sigma * model.layers[-1].trainable_variables[i]))
                    model.layers[-1].trainable_variables[i].assign(tf.sign(z) * tf.maximum(0.0, tf.abs(z) - (
                                alpha * self.optimizer.l1_regularization_strength)) / ((
                                                                                                   self.optimizer.l2_regularization_strength + tf.sqrt(
                                                                                               n + tf.square(
                                                                                                   z))) / self.optimizer.learning_rate + 1e-6))
                    n.assign_add(tf.square(grad))

                    # Update the accumulators
                    self.z_acc[i].assign(z)
                    self.n_acc[i].assign(n)

        return model

    def model_training(self, X_train, Y_train):
        model = self.current_model

        epochs = 8
        batch_size = 20
        steps_per_epoch = len(X_train) // batch_size
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = (step + 1) * batch_size
                X_batch = X_train[start:end]
                y_batch = Y_train[start:end]

                with tf.GradientTape() as tape:
                    predictions = model(X_batch)
                    loss = tf.keras.losses.mean_squared_error(y_batch, predictions)

                # Compute gradients
                gradients = tape.gradient(loss, model.layers[-1].trainable_variables)  # Exclude the last layer

                # Update the accumulators and apply the updated gradients
                for i, grad in enumerate(gradients):
                    z = self.z_acc[i]
                    n = self.n_acc[i]
                    alpha = self.optimizer.learning_rate
                    sigma = (tf.sqrt(n + tf.square(grad)) - tf.sqrt(n)) / self.alpha
                    z.assign_add(grad - (sigma * model.layers[-1].trainable_variables[i]))
                    model.layers[-1].trainable_variables[i].assign(
                        tf.sign(z) * tf.maximum(0.0, tf.abs(z) - (alpha * self.optimizer.l1_regularization_strength)) /
                        ((self.optimizer.l2_regularization_strength + tf.sqrt(n + tf.square(z))) /
                         self.optimizer.learning_rate + 1e-6)
                    )
                    n.assign_add(tf.square(grad))

                    # Update the accumulators
                    self.z_acc[i].assign(z)
                    self.n_acc[i].assign(n)

        return model

    def model_test_score(self, X_test, Y_test):
        y_test_pred = self.current_model.predict(X_test)
        score = np.sqrt(mean_squared_error(Y_test, y_test_pred))

        return score

    def get_action_len(self):
        return self.pool_component_num

    def cosine_distance(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        similarity = np.dot(a, b.T) / (a_norm * b_norm)
        dist = 1. - similarity
        return dist

    def get_pool_meta_info(self):
        pool_data_without_target = self.data_pool.drop(['target'], axis=1)
        # Cluster the data pool
        meta_gmm = GaussianMixture(n_components=self.pool_component_num, random_state=self.random_state)
        meta_gmm.fit(pool_data_without_target)
        pred_cluster_id = meta_gmm.predict(pool_data_without_target[:])

        self.data_pool['cluster_id'] = pred_cluster_id.tolist()


        # Count the number of data in each cluster
        pool_count = [0 for _ in range(self.pool_component_num)]

        group_pool = self.data_pool.groupby(['cluster_id'])
        each_cluster_ids = group_pool.size().index
        each_cluster_num = group_pool.size().values
        print('group_pool: ' + str(group_pool.size()))

        self.feature_num = self.data_pool.shape[1] - 3  # Target & Cluster_id & Selected

        for i in range(len(each_cluster_num)):
            pool_count[each_cluster_ids[i]] = each_cluster_num[i]

        self.pool_meta_info["Num of data in each cluster"] = pool_count


