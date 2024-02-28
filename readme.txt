Experiments are run on a server running Red Hat Enterprise Linux with an Intel® Xeon® CPU@2.60GHz, 512GB RAM, and two Nvidia Tesla P100 GPUs, each with 16GB of memory. We implement all algorithms in Python 3.8.

Library Requirements:
numpy == 1.19.5
sklearn == 0.0.10
pandas == 1.2.5
tensorflow-gpu == 2.4.0

Datasets:
1. CIFAR10 can be downloaded by https://www.cs.toronto.edu/~kriz/cifar.html. 
2. Other datasets are placed in folders with corresponding names. Due to the large file size of the Crop dataset, we split it into several small files. Before running experiment on Crop, need to run `cat WinnipegDataset.txt.* > WinnipegDataset.txt' first.


To run the code
1. run IAS： python IAS/main.py
2. run IAS-AMS： python IAS-AMS/main.py