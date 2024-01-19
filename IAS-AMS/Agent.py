import copy
import itertools
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



    def train_workload(self):

        time_start1 = time.time()

        while True:
            reward, done = self.env.step()
            print(reward)

            if done:
                time_end1 = time.time()

                print("The size of current training set：" + str(len(self.env.data_train)))
                print("The score of original model：" + str(self.env.original_score))
                print("The score of current model：" + str(self.env.cur_score))
                print("Benefit：" + str(self.env.original_score - self.env.cur_score))
                print("Train time: " + str(self.env.train_time))
                print("Reward time: " + str(self.env.reward_time))
                print("Time：" + str(time_end1 - time_start1))

                csv_file = open(self.mab_csv, 'a', newline='')
                csv_write = csv.writer(csv_file, dialect='excel')
                csv_write.writerow(
                    [len(self.env.data_train), self.env.original_score, self.env.cur_score,
                     self.env.original_score - self.env.cur_score,
                     self.env.train_time, time_end1 - time_start1])
                csv_file.close()


                break
