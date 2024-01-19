# coding: utf8
import copy
import itertools
import math
import random
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

    def __init__(self, data_pool, data_train, data_test, pool_component_num, batch_size, max_try_num=100,
                 random_state=0):

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

        self.cur_score = None
        self.prev_score = None

        self.original_score = None

        self.action_space = [i for i in range(self.pool_component_num)]
        self.action_valid = []

        self.train_time = 0
        self.reward_time = 0

        # Define FTRL-Proximal-like parameters
        self.optimizer = None
        self.alpha = 0.01  # Learning rate
        self.l1 = 0.1  # L1 regularization strength
        self.l2 = 0.1  # L2 regularization strength

        self.z_acc = 0
        self.n_acc = 0

        self.each_cluster_num = []

        self.eta = 0.1

        self.al = 0.1

        cluster_list = [i for i in range(self.pool_component_num)]
        self.all_coalitions = [list(j) for i in range(self.pool_component_num) for j in
                               itertools.combinations(cluster_list, r=i + 1)]
        self.action_coalitions_list = {}
        for action in range(self.pool_component_num):
            # coalitions with action
            coalitions_action = []

            # find all coalitions with action and building coalitions without action
            for c in copy.deepcopy(self.all_coalitions):
                if action in c:
                    coalitions_action.append(copy.deepcopy(c))
            self.action_coalitions_list[action] = coalitions_action

        self.ucb_list = [0 for _ in range(self.pool_component_num)]
        self.action_num_list = [0 for _ in range(self.pool_component_num)]
        self.reward_list = [0 for _ in range(self.pool_component_num)]
        self.prev_reward_list = [0 for _ in range(self.pool_component_num)]
        self.w_list = [1 for _ in range(self.pool_component_num)]
        self.prev_w_list = [1 for _ in range(self.pool_component_num)]
        self.temp_w_list = [1 for _ in range(self.pool_component_num)]
        self.m_list = [0 for _ in range(self.pool_component_num)]
        self.prev_m_list = [0 for _ in range(self.pool_component_num)]
        self.temp_m_list = [0 for _ in range(self.pool_component_num)]
        self.lambda_list = [1 for _ in range(self.pool_component_num)]

        self.init_env()

    def init_env(self):
        # 1. Init some variables
        self.data_train_history_list = []
        self.try_num = 0

        # 2. Obtain the meta-info of the data in the pool.
        # Add selected label
        self.get_pool_meta_info()
        self.data_pool['Selected'] = [0 for _ in range(len(self.data_pool))]

        group_pool = self.data_pool.groupby(['cluster_id'])
        self.each_cluster_num = group_pool.size().values

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

    def cc_shap(self, local_m, new_data_list):
        """Compute Shapley value by sampling local_m complementary contributions
        """
        n = self.pool_component_num
        local_state = np.random.RandomState(None)
        utility = np.zeros((n + 1, n))
        count = np.zeros((n + 1, n))
        idxs = np.arange(n)

        for _ in range(int(local_m)):
            local_state.shuffle(idxs)
            j = random.randint(1, n)

            new_data_action1 = []
            new_data_action2 = []
            coalition = idxs[:j]
            for id in coalition:
                new_data_action1 += [new_data_list[id]]
            data_train = pd.concat(new_data_action1).reset_index(drop=True)
            if 'cluster_id' in data_train.columns:
                data_train = data_train.drop(['cluster_id'], axis=1)

            X_train, Y_train = self.get_training_dataset(data_train)
            z = copy.deepcopy(self.z_acc)
            nn = copy.deepcopy(self.n_acc)
            variables = copy.deepcopy(self.current_model.layers[-1].trainable_variables)
            s = time.time()
            self.current_model = self.model_training(X_train, Y_train)
            e = time.time()
            self.train_time += e - s
            X_test, Y_test = self.get_test_dataset()
            u1 = self.model_test_score(X_test, Y_test)
            self.z_acc = copy.deepcopy(z)
            self.n_acc = copy.deepcopy(nn)
            for i in range(len(self.n_acc)):
                self.current_model.layers[-1].trainable_variables[i].assign(variables[i])

            coalition = idxs[j:]
            for id in coalition:
                new_data_action2 += [new_data_list[id]]
            if len(new_data_action2) != 0:
                data_train = pd.concat(new_data_action2).reset_index(drop=True)
                if 'cluster_id' in data_train.columns:
                    data_train = data_train.drop(['cluster_id'], axis=1)
                    X_train, Y_train = self.get_training_dataset(data_train)
                    z = copy.deepcopy(self.z_acc)
                    nn = copy.deepcopy(self.n_acc)
                    variables = copy.deepcopy(self.current_model.layers[-1].trainable_variables)
                    s = time.time()
                    self.current_model = self.model_training(X_train, Y_train)
                    e = time.time()
                    self.train_time += e - s
                    X_test, Y_test = self.get_test_dataset()
                    u2 = self.model_test_score(X_test, Y_test)
                    self.z_acc = copy.deepcopy(z)
                    self.n_acc = copy.deepcopy(nn)
                    for i in range(len(self.n_acc)):
                        self.current_model.layers[-1].trainable_variables[i].assign(variables[i])
            else:
                u2 = self.cur_score

            temp = np.zeros(n)
            temp[idxs[:j]] = 1
            utility[j, :] += temp * (u1 - u2)
            count[j, :] += temp

            temp = np.zeros(n)
            temp[idxs[j:]] = 1
            utility[n - j, :] += temp * (u2 - u1)
            count[n - j, :] += temp

        sv = np.zeros(n)
        for i in range(n + 1):
            for j in range(n):
                sv[j] += 0 if count[i][j] == 0 else (utility[i][j] / count[i][j])
        sv /= n
        return sv

    def update_reward(self, new_data_list):
        """
        Update acc_reward and ucb score
        """

        # traverse each cluster
        N = len(self.all_coalitions)
        m = N * 0.005
        start_time = time.time()
        sv = self.cc_shap(m, new_data_list)

        ratio = 0
        ratio_actions = {}
        for action in range(self.pool_component_num):
            ratio_action = self.action_num_list[action] / self.each_cluster_num[action]
            ratio_actions[action] = ratio_action
            ratio += ratio_action

        for action in range(self.pool_component_num):
            reward = sv[action]
            self.action_num_list[action] += 1

            # Update the aggregated score of the selected cluster
            self.prev_m_list[action] = copy.deepcopy(self.m_list[action])
            self.m_list[action] = self.m_list[action] * self.lambda_list[action] + reward
            self.prev_w_list[action] = copy.deepcopy(self.w_list[action])
            self.w_list[action] = self.w_list[action] * self.lambda_list[action] + 1
            self.prev_reward_list[action] = copy.deepcopy(self.reward_list[action])
            self.reward_list[action] = self.m_list[action] / self.w_list[action]
            self.ucb_list[action] = self.reward_list[action] + self.al * math.sqrt(
                2 * math.log(1 + ratio) / (ratio_actions[action] + 1))

            # Update lambda
            if self.try_num == 1:
                temp_m = self.prev_m_list[action]
                temp_w = self.prev_w_list[action]
                self.temp_m_list[action] = temp_m
                self.temp_w_list[action] = temp_w
            if self.try_num > 1:
                Delta_lambda = 2 * abs((self.prev_reward_list[action] - reward) * (
                        self.temp_m_list[action] - self.temp_w_list[action] * self.prev_reward_list[
                    action])) / self.prev_w_list[action]
                temp_m = self.lambda_list[action] * self.temp_m_list[action] + self.prev_m_list[
                    action]
                temp_w = self.lambda_list[action] * self.temp_w_list[action] + self.prev_w_list[
                    action]
                self.lambda_list[action] = self.lambda_list[action] - self.eta * Delta_lambda
                self.temp_m_list[action] = temp_m
                self.temp_w_list[action] = temp_w

        end_time = time.time()
        self.reward_time += end_time - start_time

    def step(self):
        """
        Execute the action
        """
        new_data = []
        new_data_list = {}
        for action in range(self.pool_component_num):
            print(f"Action:{action}")
            if self.try_num == 0:
                if len(
                        self.data_pool.loc[(self.data_pool['cluster_id'] == action) & (
                                self.data_pool['Selected'] == 0)]) >= int(self.batch_size / self.pool_component_num):
                    m = int(self.batch_size / self.pool_component_num)
                else:
                    m = len(
                        self.data_pool.loc[
                            (self.data_pool['cluster_id'] == action) & (self.data_pool['Selected'] == 0)])
            else:
                R = self.ucb_list[action]
                sum_R = sum(self.ucb_list)
                if len(
                        self.data_pool.loc[(self.data_pool['cluster_id'] == action) & (
                                self.data_pool['Selected'] == 0)]) >= int(self.batch_size * R / sum_R):
                    m = int(self.batch_size * R / sum_R)
                else:
                    m = len(
                        self.data_pool.loc[
                            (self.data_pool['cluster_id'] == action) & (self.data_pool['Selected'] == 0)])
            print('size: ' + str(m))
            self.action_num_list[action] += m
            new_data_subset = self.data_pool.loc[
                (self.data_pool['cluster_id'] == action) & (self.data_pool['Selected'] == 0)].sample(
                n=m, replace=False, random_state=self.random_state).drop(['Selected'], axis=1)
            new_data_subset_index = new_data_subset.index.to_list()
            self.data_pool.loc[new_data_subset_index, 'Selected'] = 1
            new_data += [new_data_subset]
            new_data_list[action] = new_data_subset

        # update aggregated score for each cluster

        self.update_reward(new_data_list)

        # Update the model on current training dataset

        X_train, Y_train = self.get_training_dataset(
            pd.concat(new_data).reset_index(drop=True).drop(['cluster_id'], axis=1))
        z = copy.deepcopy(self.z_acc)
        n = copy.deepcopy(self.n_acc)
        variables = copy.deepcopy(self.current_model.layers[-1].trainable_variables)
        time_start = time.time()
        self.current_model = self.model_training(X_train, Y_train)
        time_end = time.time()
        self.train_time += time_end - time_start

        X_test, Y_test = self.get_test_dataset()
        Test_score = self.model_test_score(X_test, Y_test)
        print(f"Test Score {Test_score}")

        # 5. Update the state, the reward and valid action
        reward = self.cur_score - Test_score
        if reward > 0:
            self.data_train_history_list.append(self.data_train)
            new_data.append(self.data_train)
            self.data_train = pd.concat(new_data).reset_index(drop=True)
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
                    # z += grad - (n * alpha * model.layers[-1].trainable_variables[i])
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

        # kmeans = KMeans(n_clusters=self.pool_component_num, init='k-means++', n_init=20, max_iter=1000)
        # kmeans.fit(pool_data_without_target)
        # pred_cluster_id = kmeans.predict(pool_data_without_target[:])

        self.data_pool['cluster_id'] = pred_cluster_id.tolist()

        # initial_centers = kmeans_plusplus_initializer(pool_data_without_target, self.pool_component_num).initialize()
        # metric = distance_metric(type_metric.MANHATTAN)
        # metric = distance_metric(type_metric.USER_DEFINED, func=self.cosine_distance)
        # kmeans_instance = kmeans(pool_data_without_target, initial_centers, metric=metric)
        # kmeans_instance.process()
        # clusters = kmeans_instance.get_clusters()

        # pred_cluster_id = np.array([0] * len(pool_data_without_target))
        # for i, sub in enumerate(clusters):
        # pred_cluster_id[sub] = i
        # self.data_pool['cluster_id'] = pred_cluster_id.tolist()

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
