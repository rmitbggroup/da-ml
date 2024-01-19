import numpy as np
import pandas as pd
import random


def prob_to_samplenum(total_num, prob_list):
    """
    Transfer a list of probability
    """
    sample_num_list = []
    for each_prob in prob_list:
        sample_num_list.append(round(total_num * each_prob))

    while sum(sample_num_list) < total_num:
        sample_num_list[random.randint(0, len(prob_list)) - 1] += 1

    return sample_num_list


