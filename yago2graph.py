import numpy as np
import random
import torch
import math
from random import shuffle
from DataReader import Reader


#
dataset = Reader("./yago15k/yago1992_1997_5.pkl")
# xxx = dataset.inject_anomaly()
# # print(xxx[0][1])
# # print(xxx[0][0])
# #
# xxx, y, a = dataset.get_data()
print(dataset.get_all_neighbor())
print("hhh")

print("hhh")
