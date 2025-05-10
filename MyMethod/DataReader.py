import numpy as np
import random
import torch
import math
from random import shuffle
#from transformers import  BertModel, BertTokenizer
from transformers import  BertModel, BertTokenizer
import pickle

class Reader:
    def __init__(self, path,num_neighbor=5,if_wiki=False):

        self.ent2id = dict()
        self.rel2id = dict()
        self.id2ent = dict()
        self.id2rel = dict()
        self.id2label=dict()
        self.id2description=dict()
        self.rel2label=dict()
        self.rel2description=dict()
        self.h2t = {}
        self.t2h = {}
        self.num_neighbor=num_neighbor
        self.num_anomalies = 0
        self.triples = []
        self.start_batch = 0
        self.path = path
        self.id2feature=dict()
        self.rel2feature=dict()
        self.A = {}
        #self.read_triples()
        if if_wiki:
            self.read_triples_wiki()
        else:
            self.read_triples()
        # if self.path == args.data_dir_YAGO or self.path == args.data_dir_NELL or self.path == args.data_dir_DBPEDIA:
        #     self.read_triples_yago3()
        # else:
        #     self.read_triples()
        self.labels=self.get_labels()
        self.triple_ori_set = set(self.triples)
        self.num_original_triples = len(self.triples)

        self.num_entity = self.num_ent()
        self.num_relation = self.num_rel()
        print('entity&relation: ', self.num_entity, self.num_relation)

        #self.bp_triples_label = self.inject_anomaly(args)
        #self.t_triples_label=self.combine_label()
        self.num_triples_with_anomalies = len(self.triples)
        self.train_data, self.labels = self.get_data()
        #self.triples_with_anomalies, self.triples_with_anomalies_labels = self.get_data_test()

    # def train_triples(self):
    #     return self.triples["train"]
    #
    # def valid_triples(self):
    #     return self.triples["valid"]
    #
    # def test_triples(self):
    #     return self.triples["test"]

    # def all_triples(self):
    #     return self.triples["train"] + self.triples["valid"] + self.triples["test"]

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def get_add_ent_id(self, ent):
        if ent in self.ent2id:
            ent_id = self.ent2id[ent]
        else:
            ent_id = len(self.ent2id)
            self.ent2id[ent] = ent_id
            self.id2ent[ent_id] = ent

        return ent_id

    def get_add_wiki_ent_id(self, ent,label,description,embedding):

        if ent in self.ent2id:
            ent_id = self.ent2id[ent]
        else:
            ent_id = len(self.ent2id)
            self.ent2id[ent] = ent_id
            self.id2ent[ent_id] = ent
            self.id2label[ent_id]=label
            self.id2description[ent_id]=description
            self.id2feature[ent_id]=embedding
            #self.id2feature[ent_id]=哈哈哈哈
        return ent_id
    def get_add_rel_id(self, rel):
        if rel in self.rel2id:
            rel_id = self.rel2id[rel]
        else:
            rel_id = len(self.rel2id)
            self.rel2id[rel] = rel_id
            self.id2rel[rel_id] = rel
        return rel_id

    def get_add_wiki_rel_id(self, rel,label,description,embedding):
        if rel in self.rel2id:
            rel_id = self.rel2id[rel]
        else:
            rel_id = len(self.rel2id)
            self.rel2id[rel] = rel_id
            self.id2rel[rel_id] = rel
            self.rel2label[rel_id] = label
            self.rel2description[rel_id] = description
            self.rel2feature[rel_id]=embedding
        return rel_id

    def init_embeddings(self, entity_file, relation_file):
        entity_emb, relation_emb = [], []

        with open(entity_file) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])

        with open(relation_file) as f:
            for line in f:
                relation_emb.append([float(val) for val in line.strip().split()])

        return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

    def read_triples(self):
        print('Read begin!')
        #----------------------------------------
        with open(self.path, "rb") as file:
            loaded_list = pickle.load(file)
            #print(loaded_list)
            for list in loaded_list:
                print(list[9])
                head=list[0]
                rel=list[1]
                tail=list[2]
                label=list[5]
                head_id = self.get_add_ent_id(head)
                rel_id = self.get_add_rel_id(rel)
                tail_id = self.get_add_ent_id(tail)

                self.triples.append((head_id, rel_id, tail_id,label))

                self.A[(head_id, tail_id)] = rel_id
                # self.A[head_id][tail_id] = rel_id

                # generate h2t
                if not head_id in self.h2t.keys():
                    self.h2t[head_id] = set()
                temp = self.h2t[head_id]
                temp.add(tail_id)
                self.h2t[head_id] = temp

                # generate t2h
                if not tail_id in self.t2h.keys():
                    self.t2h[tail_id] = set()
                temp = self.t2h[tail_id]
                temp.add(head_id)
                self.t2h[tail_id] = temp
        #-------------------------------------------
        # for file in ["train", "valid", "test"]:
        #     with open(self.path + '/' + file + ".txt", "r") as f:
        #         for line in f.readlines():
        #             try:
        #                 head, rel, tail = line.strip().split("\t")
        #             except:
        #                 print(line)
        #             head_id = self.get_add_ent_id(head)
        #             rel_id = self.get_add_rel_id(rel)
        #             tail_id = self.get_add_ent_id(tail)
        #
        #             self.triples.append((head_id, rel_id, tail_id))
        #
        #             self.A[(head_id, tail_id)] = rel_id
        #             # self.A[head_id][tail_id] = rel_id
        #
        #             # generate h2t
        #             if not head_id in self.h2t.keys():
        #                 self.h2t[head_id] = set()
        #             temp = self.h2t[head_id]
        #             temp.add(tail_id)
        #             self.h2t[head_id] = temp
        #
        #             # generate t2h
        #             if not tail_id in self.t2h.keys():
        #                 self.t2h[tail_id] = set()
        #             temp = self.t2h[tail_id]
        #             temp.add(head_id)
        #             self.t2h[tail_id] = temp
        print("Read end!")
        return self.triples

    def read_triples_wiki(self):
        print('Read begin!')
        #----------------------------------------
        with open(self.path, "rb") as file:
            loaded_list = pickle.load(file)
            #print(loaded_list)
            for list in loaded_list:
                head=list[0]
                rel=list[1]
                tail=list[2]
                head_label=list[3]
                head_description=list[4]
                relation_label=list[5]
                relation_description=list[6]
                tail_label=list[7]
                tail_description=list[8]
                label=list[9]
                head_embedding=list[10]
                relation_embedding=list[11]
                tail_embedding=list[12]
                head_id = self.get_add_wiki_ent_id(head,head_label,head_description,head_embedding)
                rel_id = self.get_add_wiki_rel_id(rel,relation_label,relation_description,relation_embedding)
                tail_id = self.get_add_wiki_ent_id(tail,tail_label,tail_description,tail_embedding)

                self.triples.append((head_id, rel_id, tail_id,label))

                self.A[(head_id, tail_id)] = rel_id
                # self.A[head_id][tail_id] = rel_id

                # generate h2t
                if not head_id in self.h2t.keys():
                    self.h2t[head_id] = set()
                temp = self.h2t[head_id]
                temp.add(tail_id)
                self.h2t[head_id] = temp

                # generate t2h
                if not tail_id in self.t2h.keys():
                    self.t2h[tail_id] = set()
                temp = self.t2h[tail_id]
                temp.add(head_id)
                self.t2h[tail_id] = temp

        print("Read end!")
        return self.triples
    def read_triples_yago3(self):
        print('Read begin!')
        for file in ["train", "valid", "test"]:
            with open(self.path + '/' + file + ".txt", "r", encoding="utf-8") as f:
                train = f.readlines()
                # train_ = set({})
                for i in range(len(train)):
                    x = train[i].split()
                    x_ = tuple(x)
                    head, rel, tail = x_[0], x_[1], x_[2]

                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)

                    self.triples.append((head_id, rel_id, tail_id))

                    self.A[(head_id, tail_id)] = rel_id
                    # self.A[head_id][tail_id] = rel_id

                    # generate h2t
                    if not head_id in self.h2t.keys():
                        self.h2t[head_id] = set()
                    temp = self.h2t[head_id]
                    temp.add(tail_id)
                    self.h2t[head_id] = temp

                    # generate t2h
                    if not tail_id in self.t2h.keys():
                        self.t2h[tail_id] = set()
                    temp = self.t2h[tail_id]
                    temp.add(head_id)
                    self.t2h[tail_id] = temp

                del (train)

        print("Read end!")
        return self.triples

    def rand_ent_except(self, ent):
        rand_ent = random.randint(1, self.num_ent() - 1)
        while rand_ent == ent:
            rand_ent = random.randint(1, self.num_ent() - 1)
        return rand_ent

    def generate_neg_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 1)
            if head_or_tail == 0:
                new_head = self.rand_ent_except(head)
                neg_triples.append((new_head, rel, tail))
            else:
                new_tail = self.rand_ent_except(tail)
                neg_triples.append((head, rel, new_tail))
        return neg_triples

    def generate_anomalous_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 2)
            if head_or_tail == 0:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = rel
                new_tail = tail
                # neg_triples.append((new_head, rel, tail))
            elif head_or_tail == 1:
                new_head = head
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = tail
            else:
                # new_tail = self.rand_ent_except(tail)
                # neg_triples.append((head, rel, new_tail))
                new_head = head
                new_relation = rel
                new_tail = random.randint(0, self.num_entity - 1)
            anomaly = (new_head, new_relation, new_tail)
            while anomaly in self.triple_ori_set:
                if head_or_tail == 0:
                    new_head = random.randint(0, self.num_entity - 1)
                    new_relation = rel
                    new_tail = tail
                    # neg_triples.append((new_head, rel, tail))
                elif head_or_tail == 1:
                    new_head = head
                    new_relation = random.randint(0, self.num_relation - 1)
                    new_tail = tail
                else:
                    # new_tail = self.rand_ent_except(tail)
                    # neg_triples.append((head, rel, new_tail))
                    new_head = head
                    new_relation = rel
                    new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)
            neg_triples.append(anomaly)
        return neg_triples

    def generate_anomalous_triples_2(self, num_anomaly):
        neg_triples = []
        for i in range(num_anomaly):
            new_head = random.randint(0, self.num_entity - 1)
            new_relation = random.randint(0, self.num_relation - 1)
            new_tail = random.randint(0, self.num_entity - 1)

            anomaly = (new_head, new_relation, new_tail)

            while anomaly in self.triple_ori_set:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)

            neg_triples.append(anomaly)
        return neg_triples

    def shred_triples(self, triples):
        h_dix = [triples[i][0] for i in range(len(triples))]
        r_idx = [triples[i][1] for i in range(len(triples))]
        t_idx = [triples[i][2] for i in range(len(triples))]
        return h_dix, r_idx, t_idx

    def shred_triples_and_labels(self, triples_and_labels):
        heads = [triples_and_labels[i][0][0] for i in range(len(triples_and_labels))]
        rels = [triples_and_labels[i][0][1] for i in range(len(triples_and_labels))]
        tails = [triples_and_labels[i][0][2] for i in range(len(triples_and_labels))]
        labels = [triples_and_labels[i][1] for i in range(len(triples_and_labels))]
        return heads, rels, tails, labels

    def all_triplets(self):
        ph_all, pr_all, pt_all = self.shred_triples(self.triples)
        nh_all, nr_all, nt_all = self.shred_triples(self.generate_neg_triples(self.triples))
        return ph_all, pt_all, nh_all, nt_all, pr_all

    def get_data(self):
        # bp_triples_label = self.inject_anomaly()
        #bp_triples_label = self.bp_triples_label
        bp_triples_label=self.triples
        labels = [bp_triples_label[i][3] for i in range(len(bp_triples_label))]
        bp_triples = [(bp_triples_label[i][0],bp_triples_label[i][1],bp_triples_label[i][2]) for i in range(len(bp_triples_label))]
        bn_triples = self.generate_anomalous_triples(bp_triples)
        all_triples = bp_triples + bn_triples

        return self.toarray(all_triples), self.toarray(labels)
    def get_labels(self):
        labels=[]
        for t in self.triples:
            labels.append(t[3])
        return labels
    def get_data_test(self):
        bp_triples_label = self.bp_triples_label
        labels = [bp_triples_label[i][1] for i in range(len(bp_triples_label))]
        bp_triples = [bp_triples_label[i][0] for i in range(len(bp_triples_label))]

        return self.toarray(bp_triples), self.toarray(labels)

    def toarray(self, x):
        return torch.from_numpy(np.array(list(x)).astype(np.int32))

    def inject_anomaly(self, args):
        print("Inject anomalies!")
        original_triples = self.triples
        triple_size = len(original_triples)

        self.num_anomalies = int(args.anomaly_ratio * self.num_original_triples)
        args.num_anomaly_num = self.num_anomalies
        print("###########Inject TOP@K% Anomalies##########")
        # if self.isInjectTopK:
        #     self.num_anomalies = args.num_anomaly_num
        #     print("###########Inject TOP@K Anomalies##########")
        # else:
        #

        # idx = random.sample(range(0, self.num_original_triples - 1), num_anomalies)
        idx = random.sample(range(0, self.num_original_triples - 1), self.num_anomalies // 2)
        selected_triples = [original_triples[idx[i]] for i in range(len(idx))]
        anomalies = self.generate_anomalous_triples(selected_triples) + self.generate_anomalous_triples_2(self.num_anomalies // 2)

        triple_label = [(original_triples[i], 0) for i in range(len(original_triples))]
        anomaly_label = [(anomalies[i], 1) for i in range(len(anomalies))]

        triple_anomaly_label = triple_label + anomaly_label
        shuffle(triple_anomaly_label)
        return triple_anomaly_label
    def combine_label(self):
        original_triples = self.triples
        with open(self.path, "rb") as file:
            loaded_list = pickle.load(file)
            #triple_label = [(original_triples[i], j) for i,j in range(len(original_triples)),loaded_list]
            triple_label = [
                ((original_triples[i][0], original_triples[i][1], original_triples[i][2]), loaded_list[i][5]) for i in range(len(original_triples))]
            return triple_label

    def get_neighbor_id(self,ent):
        hrt = []
        h2t=self.h2t
        t2h=self.t2h
        A=self.A
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

    def get_triple_neighbor(self,h, r, t):
        num_neighbor=self.num_neighbor
        h2t = self.h2t
        t2h = self.t2h
        A = self.A

        # print('h, r, t', h, r, t)
        head_neighbor = self.get_neighbor_id(h)
        tail_neighbor = self.get_neighbor_id(t)
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
    def get_all_neighbor(self):
        all_neighbor=[]
        for i in range(len(self.triples)):
            hrt_neighbor = self.get_triple_neighbor(self.triples[i][0], self.triples[i][1], self.triples[i][2])
            all_neighbor.append(hrt_neighbor)
        return all_neighbor

    def bert_encoder(self,text):
        # 加载预训练的分词器

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 加载预训练的BERT模型
        model = BertModel.from_pretrained('bert-base-uncased')

        # 使用分词器对文本进行处理
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # 获取BERT模型的输出
        outputs = model(**inputs)
        cls=outputs.last_hidden_state[:, 0, :]
        return cls[0]

            #print(i)
            #print(hrt_neighbor)
            #print('11')
# dataset = Reader(args.data_dir_FB, "train")
# xxx = dataset.inject_anomaly()
# # print(xxx[0][1])
# # print(xxx[0][0])
# #
# xxx, y, a = dataset.get_data()
# print(xxx[0])
