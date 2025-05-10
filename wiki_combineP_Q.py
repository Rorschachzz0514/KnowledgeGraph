import numpy as np
import random
import torch
import math
from random import shuffle
from DataReader import Reader
import pickle
from transformers import  BertModel, BertTokenizer
begin=1975
endding=1980
with open("./wiki/wiki"+str(begin)+"_"+str(endding)+"_5.pkl", "rb") as file:
    data = pickle.load(file)
with open("./wiki/P_information", "rb") as file:
    P_information = pickle.load(file)
# with open("wiki/Q_information0_100.pkl", "rb") as file:
#     Q_information0_100 = pickle.load(file)
# with open("wiki/Q_information101_120.pkl", "rb") as file:
#     Q_information101_120 = pickle.load(file)
# with open("wiki/Q_information121_201.pkl", "rb") as file:
#     Q_information121_201 = pickle.load(file)
# with open("wiki/Q_information202_300.pkl", "rb") as file:
#     Q_information202_300 = pickle.load(file)
# with open("wiki/Q_information301_401.pkl", "rb") as file:
#     Q_information301_401 = pickle.load(file)
# with open("wiki/Q_information402_502.pkl", "rb") as file:
#     Q_information402_502 = pickle.load(file)
# with open("wiki/Q_information503_603.pkl", "rb") as file:
#     Q_information503_603 = pickle.load(file)
# with open("wiki/Q_information604_704.pkl", "rb") as file:
#     Q_information604_704 = pickle.load(file)
# with open("wiki/Q_information705_805.pkl", "rb") as file:
#     Q_information705_805 = pickle.load(file)
# with open("wiki/Q_information806_906.pkl", "rb") as file:
#     Q_information806_906 = pickle.load(file)
# with open("wiki/Q_information907_1007.pkl", "rb") as file:
#     Q_information907_1007 = pickle.load(file)
# with open("wiki/Q_information1008_1108.pkl", "rb") as file:
#     Q_information1008_1108 = pickle.load(file)
#

def bert_encoder(text):
    # 加载预训练的分词器
    #本地运行-------------------------------
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # # 加载预训练的BERT模型
    # model = BertModel.from_pretrained('bert-base-uncased')
    # save_directory = "./LLM_local_model"
    # tokenizer.save_pretrained(save_directory)
    # model.save_pretrained(save_directory)
    #本地运行--------------------------------------
    #服务器运行---------------------------------
    # 从本地文件夹加载分词器和模型
    load_directory = "./LLM_local_model"

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(load_directory)

    # 加载模型
    model = BertModel.from_pretrained(load_directory)

    #服务器运行---------------------------
    # 使用分词器对文本进行处理
    inputs = tokenizer(text, return_tensors='pt', padding="max_length", truncation=True, max_length=21)

    # 获取BERT模型的输出
    outputs = model(**inputs)
    #cls=outputs.last_hidden_state[:, 0, :]
    return outputs.last_hidden_state,inputs['attention_mask']

ranges = [
    (0, 200),
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1000),
    (1000, 1200),
    (1200, 1400),
    (1400, 1600),
    (1600, 1800),
    (1800, 2000),
    (2000, 2200),
    (2200, 2400),
    (2400, 2600),
    (2600, 2800),
    (2800, 3000),
    (3000, 3200),
    (3200, 3400),
    (3400, 3600),
    (3600, 3800),
    (3800, 4000),
    (4000, 4200),
    (4200, 4400),
    (4400, 4600),
    (4600, 4800),
    (4800, 5000),
    (5000, 5200),
    (5200, 5400),
    (5400, 5600),
    (5600, 5800),
    (5800, 6000),
    (6000, 6200),
    (6200, 6400),
    (6400, 6600),
    (6600, 6800),
    (6800, 7000),
    (7000, 7200),
    (7200, 7400),
    (7400, 7600),
    (7600, 7800),
    (7800, 8000),
    (8000, 8200),
    (8200, 8458),

]
Q_information=[]
Q_No_information=[]
# 生成代码
#---------------------------有数据的
for start, end in ranges:
    yes_file_name = f"Q_information{start}_{end}.pkl"
    with open("wiki/"+yes_file_name, "rb") as file:
        temp_yes=pickle.load(file)
        Q_information=Q_information+temp_yes
    no_file_name = f"Q_No_information{start}_{end}.pkl"
    with open("wiki/" + no_file_name, "rb") as file:
        temp_no = pickle.load(file)
        Q_No_information = Q_No_information + temp_no
    print("start",start,"end",end)
    print("yeslength",len(temp_yes),"nolength",len(temp_no),)
    print(end-start+1,len(temp_yes)+len(temp_no))
#--------------------------有数据的

#--------------------------无数据的
# for start, end in ranges:
#     file_name = f"Q_No_information{start}_{end}.pkl"
#     with open("wiki/" + file_name, "rb") as file:
#         temp = pickle.load(file)
#         Q_information = Q_information + temp
#--------------------无数据的
    # variable_name = f"Q_information{start}_{end}"
    # code = f'with open("wiki/{file_name}", "rb") as file:\n'
    # code += f'    {variable_name} = pickle.load(file)\n'
    # print(code)
# with open("./wiki/Q_information0_8457.pkl","wb") as file:
#     pickle.dump(Q_information, file)
#dataset = Reader("./wiki/Q_information0_100.pkl.pkl")
single_data_lable_description=[]
i=0
for d in data:
    print(i)
    i=i+1
    temp=[]
    Q_head_count=0
    Q_tail_count=0
    P_count=0
    temp.append(d[0])
    temp.append(d[1])
    temp.append(d[2])
    for Q in Q_information:
        if d[0]==Q[0]:#找到了
            Q_head_count=Q_head_count+1
            temp.append(Q[1])
            temp.append(Q[2])
            break
    if Q_head_count==0:##没找到label和description

        temp.append(d[0])
        temp.append(d[0])

    for P in P_information:
        if d[1]==P[0]:
            P_count=P_count+1
            temp.append(P[1])
            temp.append(P[2])
            break
    if P_count==0:
        temp.append(d[1])
        temp.append(d[1])

    for Q in Q_information:
        if d[2]==Q[0]:
            Q_tail_count=Q_tail_count+1
            temp.append(Q[1])
            temp.append(Q[2])
            break
    if Q_tail_count==0:
        temp.append(d[2])
        temp.append(d[2])
    #print(d[5])
    #print(temp)
    temp.append(d[5])
    head_text_embedding,head_mask=bert_encoder("label:"+temp[3]+',description:'+temp[4])
    re_text_embedding,re_mask=bert_encoder("label:"+temp[5]+',description:'+temp[6])
    tail_text_embedding,tail_mask=bert_encoder("label:"+temp[7]+',description:'+temp[8])
    #attention_mask = inputs['attention_mask']
    temp.append([head_text_embedding,head_mask])
    temp.append([re_text_embedding,re_mask])
    temp.append([tail_text_embedding,tail_mask])
    #print(temp[9])
    single_data_lable_description.append(temp)
with open("./wiki/single_data_label_description_"+str(begin)+"_"+str(endding)+".pkl","wb") as file:
    pickle.dump(single_data_lable_description, file)
# xxx = dataset.inject_anomaly()
# # print(xxx[0][1])
# # print(xxx[0][0])
# #
# xxx, y, a = dataset.get_data()


print("hhh")
