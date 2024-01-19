import copy
import math
import random

import numpy
import torch
from sklearn.mixture import GaussianMixture
import time
import csv
import pandas as pd


class Data_agent(object):
    """
    The agent of AutoData
    """

    def __init__(self, env, mabcsv, random_state):
        self.env = env
        self.mab_csv = mabcsv
        self.random_state = random_state
        self.alpha = 0.1
        self.eta = 0.1

        self.ucb_list = [0 for _ in range(self.env.pool_component_num)]
        self.action_num_list = [0 for _ in range(self.env.pool_component_num)]
        self.action_k_list = [[] for _ in range(self.env.pool_component_num)]
        self.reward_list = [0 for _ in range(self.env.pool_component_num)]
        self.prev_reward_list = [0 for _ in range(self.env.pool_component_num)]
        self.w_list = [1 for _ in range(self.env.pool_component_num)]
        self.prev_w_list = [1 for _ in range(self.env.pool_component_num)]
        self.temp_w_list = [1 for _ in range(self.env.pool_component_num)]
        self.m_list = [0 for _ in range(self.env.pool_component_num)]
        self.prev_m_list = [0 for _ in range(self.env.pool_component_num)]
        self.temp_m_list = [0 for _ in range(self.env.pool_component_num)]
        self.lambda_list = [1 for _ in range(self.env.pool_component_num)]
        self.prev_lambda_list = [1 for _ in range(self.env.pool_component_num)]
        self.k_last_list = [0 for _ in range(self.env.pool_component_num)]

        self.clusters_info = []
        self.clusters_neighbor = []
        self.clusters_distance = []

        self.max_cluster_distance = 0

        self.get_pool_cluster_info()

    def get_pool_cluster_info(self):
        group_pool = self.env.data_pool.groupby(['cluster_id'])
        each_cluster_ids = group_pool.size().index
        each_cluster_num = group_pool.size().values

        # Get the information (mu and sigma) of each cluster
        for i in range(len(each_cluster_num)):
            if each_cluster_num[i] > 0:
                gmm_one_cluster = GaussianMixture(n_components=1, random_state=self.random_state,
                                                  covariance_type='full')
                gmm_one_cluster.fit(self.env.data_pool[self.env.data_pool['cluster_id'] == each_cluster_ids[i]].drop(
                    ['target', 'cluster_id', 'Selected'], axis=1))

                cluster_means = gmm_one_cluster.means_[0]
                cluster_convariance = gmm_one_cluster.covariances_[0]

                self.clusters_info.append([cluster_means, cluster_convariance, each_cluster_num[i]])

        # Get the distance between clusters
        for i in range(len(each_cluster_num)):
            i_dist_list = []
            for j in range(len(each_cluster_num)):
                if j != i:
                    i_j_dist = self.Wasserstein_dist(self.clusters_info[i][0], self.clusters_info[j][0],
                                                     self.clusters_info[i][1], self.clusters_info[j][1])
                    i_dist_list.append(i_j_dist)

                    if i_j_dist > self.max_cluster_distance:
                        self.max_cluster_distance = i_j_dist
                else:
                    i_dist_list.append(0)

            self.clusters_distance.append(i_dist_list)

    def Wasserstein_dist(self, mu_1, mu_2, sigma_1, sigma_2):
        """
        Compute the Wasserstein distance of two distribution
        """
        p1 = torch.sum(torch.pow((torch.from_numpy(mu_1) - torch.from_numpy(mu_2)), 2), 0)
        p2 = torch.sum(torch.pow(
            torch.pow(torch.abs(torch.from_numpy(sigma_1)), 0.5) - torch.pow(torch.abs(torch.from_numpy(sigma_2)), 0.5),
            2), 1)
        dist = p1 + torch.sum(p2, 0)

        return dist.item()

    def choose_action(self):
        """
        Choose a new action
        """
        if self.ucb_list.count(max(self.ucb_list)) > 1:
            max_ucb_score = max(self.ucb_list)
            max_index = []
            for i in range(0, len(self.ucb_list)):
                if self.ucb_list[i] == max_ucb_score:
                    max_index.append(i)

            action = max_index[random.randint(0, len(max_index) - 1)]
        else:
            action = self.ucb_list.index(max(self.ucb_list))

        return action

    def update_ucb(self, action_index, step_reward):
        """
        Update acc_reward and ucb score
        """

        self.action_num_list[action_index] += 1

        # Update the aggregated score of the selected cluster
        self.prev_m_list[action_index] = copy.deepcopy(self.m_list[action_index])
        self.m_list[action_index] = self.m_list[action_index] * self.lambda_list[action_index] + step_reward
        self.prev_w_list[action_index] = copy.deepcopy(self.w_list[action_index])
        self.w_list[action_index] = self.w_list[action_index] * self.lambda_list[action_index] + 1
        self.prev_reward_list[action_index] = copy.deepcopy(self.reward_list[action_index])
        self.reward_list[action_index] = self.m_list[action_index] / self.w_list[action_index]

        # Update the ucb of the selected cluster
        num = 0
        for i in self.action_k_list[action_index]:
            num += (self.env.try_num - i + 1) / self.env.try_num
        self.ucb_list[action_index] = self.reward_list[action_index] + self.alpha * math.sqrt(2 * math.log(sum(self.action_num_list))/(num+1))


        # Update lambda
        if self.action_num_list[action_index] == 2:
            temp_m = self.prev_m_list[action_index]
            temp_w = self.prev_w_list[action_index]
            self.temp_m_list[action_index] = temp_m
            self.temp_w_list[action_index] = temp_w
        if self.action_num_list[action_index] > 2:
            Delta_lambda = 2 * abs((self.prev_reward_list[action_index] - step_reward) * (
                    self.temp_m_list[action_index] - self.temp_w_list[action_index] * self.prev_reward_list[action_index])) / self.prev_w_list[action_index]
            temp_m = self.lambda_list[action_index] * self.temp_m_list[action_index] + self.prev_m_list[action_index]
            temp_w = self.lambda_list[action_index] * self.temp_w_list[action_index] + self.prev_w_list[
                action_index]
            self.lambda_list[action_index] = self.lambda_list[action_index] - self.eta * Delta_lambda
            self.temp_m_list[action_index] = temp_m
            self.temp_w_list[action_index] = temp_w

        # Update the ucb of other clusters
        for i in range(self.env.pool_component_num):
            if i != action_index:
                self.prev_m_list[i] = copy.deepcopy(self.m_list[i])
                self.m_list[i] = (self.env.try_num - self.k_last_list[i]) / self.env.pool_component_num * \
                                      self.m_list[i] * self.lambda_list[
                                          i]
                self.prev_w_list[i] = copy.deepcopy(self.w_list[i])
                self.w_list[i] = (self.env.try_num - self.k_last_list[i]) / self.env.pool_component_num * self.w_list[
                    i] * self.lambda_list[
                                     i]
                self.prev_reward_list[i] = copy.deepcopy(self.reward_list[i])
                self.reward_list[i] = self.m_list[i] / self.w_list[i]
                num = 0
                for j in self.action_k_list[i]:
                    num += (self.env.try_num - j + 1) / self.env.try_num
                self.ucb_list[i] = self.reward_list[i] + (self.alpha + self.clusters_distance[i][action_index]/self.max_cluster_distance) * math.sqrt(
                    2 * math.log(sum(self.action_num_list)) / (num + 1))

    def train_workload(self):

        time_start1 = time.time()

        while True:
            action = 0
            if self.env.try_num == 0:
                # sample the cluster away from T_train
                gmm_T_train = GaussianMixture(n_components=1, random_state=self.random_state,
                                              covariance_type='full')
                gmm_T_train.fit(self.env.data_train.drop(['target'], axis=1))

                T_train_means = gmm_T_train.means_[0]
                T_train_convariance = gmm_T_train.covariances_[0]
                maxdist = 0
                for i in range(self.env.pool_component_num):
                    dist = self.Wasserstein_dist(self.clusters_info[i][0], T_train_means, self.clusters_info[i][1],
                                                 T_train_convariance)
                    if dist > maxdist:
                        action = i
                        maxdist = dist
            else:
                action = self.choose_action()
            reward, done = self.env.step(action)
            print(reward)
            if done == 'null':
                continue
            self.k_last_list[action] = self.env.try_num
            self.action_k_list[action] += [self.env.try_num]
            self.update_ucb(action, reward)

            if done:
                time_end1 = time.time()
                print("The size of current training set：" + str(len(self.env.data_train)))
                print("The score of original model：" + str(self.env.original_score))
                print("The score of current model：" + str(self.env.cur_score))
                print("Benefit：" + str(self.env.original_score - self.env.cur_score))
                print("Train time: " + str(self.env.train_time))
                print("Time：" + str(time_end1 - time_start1))

                csv_file = open(self.mab_csv, 'a', newline='')
                csv_write = csv.writer(csv_file, dialect='excel')
                csv_write.writerow(
                    [len(self.env.data_train), self.env.original_score, self.env.cur_score,
                     self.env.original_score - self.env.cur_score,
                     self.env.train_time, time_end1 - time_start1])
                csv_file.close()


                break
