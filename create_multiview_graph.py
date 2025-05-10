import torch
import numpy as np
import random
from DataReader import Reader

def get_neighbor_id(ent, h2t, t2h, A):
    hrt = []
    hrt1 = []
    hrt2 = []
    if ent in h2t.keys():
        tails = list(h2t[ent])
        # print('tails', tails)
        # for i in range(len(tails)):
        #     hrt1.append((ent, A[ent][tails[i]], tails[i]))
        hrt1 = [(ent, A[(ent, i)], i) for i in tails]

        # print('hrt1', hrt1)

    if ent in t2h.keys():
        heads = list(t2h[ent])
        # for i in range(len(heads)):
        #     hrt2.append((heads[i], A[heads[i]][ent], ent))
        hrt2 = [(i, A[(i, ent)], ent) for i in heads]
        # print('hrt2', hrt2)

    hrt = hrt1 + hrt2

    return hrt
def get_triple_neighbor(h, r, t, dataset, num_neighbor):
    h_neighbor = 0
    h2t = dataset.h2t
    t2h = dataset.t2h
    A = dataset.A

    # print('h, r, t', h, r, t)
    head_neighbor = get_neighbor_id(h, h2t, t2h, A)
    tail_neighbor = get_neighbor_id(t, h2t, t2h, A)
    # hrt_neighbor = get_neighbor_id(h, h2t, t2h, A) + get_neighbor_id(t, h2t, t2h, A)
    # while (h, r, t) in hrt_neighbor:
    #     hrt_neighbor.remove((h, r, t))
    # if len(head_neighbor) == 0:
    #     temp = [(h, r, t)]
    #     hh_neighbors = random.choices(temp, k=num_neighbor)
    if len(head_neighbor) > num_neighbor:
        hh_neighbors = random.sample(head_neighbor, k=num_neighbor)
    elif len(head_neighbor) > 0:
        hh_neighbors = random.choices(head_neighbor, k=num_neighbor)
    else:
        temp = [(h, r, t)]
        hh_neighbors = random.choices(temp, k=num_neighbor)

    if len(tail_neighbor) > num_neighbor:
        tt_neighbors = random.sample(tail_neighbor, k=num_neighbor)
    elif num_neighbor > len(tail_neighbor):
        if len(tail_neighbor) > 0:
            tt_neighbors = random.choices(tail_neighbor, k=num_neighbor)
        else:
            # print('hrt=null', len(hrt_neighbor))
            temp = [(h, r, t)]
            tt_neighbors = random.choices(temp, k=num_neighbor)
    else:
        tt_neighbors = tail_neighbor

    hrt_neighbor = [(h, r, t)] + hh_neighbors + [(h, r, t)] + tt_neighbors
    # print('hrt_neighbor', len(hrt_neighbor))
    # print("hn, tn", (len(head_neighbor), len(tail_neighbor)))


    return hrt_neighbor


dataset = Reader("./wiki/single_data_label_description.pkl",True)
sample_triple=dataset.triples
num_neighbor=10
for i in range(len(sample_triple)):
    hrt_neighbor = get_triple_neighbor(sample_triple[i][0], sample_triple[i][1],sample_triple[i][2], dataset, num_neighbor)
    print(hrt_neighbor)
print('hhh')