import pandas as pd

from Environment import Data_env
from Agent import Data_agent


from sys import argv

# Parameters for the environment
for i in range(20, 21):
    dataset = 'Traffic'
    data_pool_path = "../data/" + dataset + "/pool.csv"
    data_train_path = "../data/" + dataset + "/train.csv"
    data_test_path = "../data/" + dataset + "/test.csv"
    pool_cluster_num = 10
    max_try_num = i

    print('# of itration: ' + str(max_try_num))


    random_state = 40

    data_pool = pd.read_csv(data_pool_path)
    data_train = pd.read_csv(data_train_path)
    data_test = pd.read_csv(data_test_path)
    train_size = data_train.shape[0]
    print('initial train dataset size: ' + str(train_size))
    batch_size = int(train_size / 5)

    env = Data_env(data_pool, data_train, data_test,
                       pool_cluster_num, batch_size, max_try_num)

    # Parameters for the agent

    train_csv = "../" + dataset + "/IAS/res_" + str(
        max_try_num) + ".csv"

    autodata = Data_agent(env, train_csv, random_state)

    print("Agent Ready!")

    # Train the workload
    autodata.train_workload()