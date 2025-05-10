import numpy as np
import random
import torch
import math
from random import shuffle
from DataReader import Reader


#
dataset = Reader("./wiki/single_data_label_description.pkl",True)
# xxx = dataset.inject_anomaly()
# # print(xxx[0][1])
# # print(xxx[0][0])
# #
# xxx, y, a = dataset.get_data()
print(dataset.get_all_neighbor())
print("hhh")
