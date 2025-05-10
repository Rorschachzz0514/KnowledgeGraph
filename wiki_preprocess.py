import re
import csv
import pandas as pd
import pickle

train_path="../mmkb-master/TemporalKGs/wikidata/wiki_train.txt"
test_path="../mmkb-master/TemporalKGs/wikidata/wiki_test.txt"
valid_path="../mmkb-master/TemporalKGs/wikidata/wiki_valid.txt"
data=[]
num=0
#1删除无时间数据
with open(train_path, 'r', encoding='utf-8') as file:
    for line in file:
        #print(line.strip())  # 使用 strip() 去掉每行末尾的换行符
        #data.append(line)
        #if '##' in line:
        #print(line)
        data.append(line)
with open(test_path, 'r', encoding='utf-8') as file:
    for line in file:
        #print(line.strip())  # 使用 strip() 去掉每行末尾的换行符
        #data.append(line)
        #if '##' in line:
            #print(line)
        data.append(line)
with open(valid_path, 'r', encoding='utf-8') as file:
    for line in file:
        #print(line.strip())  # 使用 strip() 去掉每行末尾的换行符
        #data.append(line)
        #if '##' in line:
        #print(line)
        data.append(line)
#此时的data是 'Q430543\tP1087\t2257\toccurUntil\t2015\n',需要取出data行后的回车
for i in range(0,len(data)):
    data[i]=data[i].strip()
#print(data)
#'Q164536\tP1082\t2648\toccurSince\t1869',已经删除回车，需要通过\t将所有数据切分
for i in range(0,len(data)):
    data[i]=data[i].split("\t")
#print(data)
#['Q486592', 'P1087', '2541', 'occurSince', '2009'],此时，每一个数据都已经切分，下面需要时期配对，要求每个数据有起始时间和结束时间
# wiki 与yago不一样，配对的信息并不是相邻的，需要通篇寻找配对信息，（其实yago也应该这么做）
# 因为下面要寻找配对信息，所以要给每一个data加标识符，1表示已经配对成功，0.表示还没有进行配对
for i in range(0,len(data)):
    data[i].append(0)
#print(data)
double_data=[]

#设计双层循环寻找配对信息化
#----------------------这个代码有用，但是跑的时间太长了，所以我跑了一遍，结果保存到本地，wiki.pkl
# for i in range(0,len(data)-1):
#     if i % 1000 == 0:
#         print(i)
#         print(i/len(data))
#     for j in range(i+1,len(data)-1):
#         if data[j][0]==data[i][0] and data[j][1]==data[i][1] and data[j][2]==data[i][2] and data[i][-1]==0 and data[j][-1]==0:
#             if data[i][3]=='occurSince' and data[j][3]=='occurUntil':
#                 data[i][-1]=1
#                 data[j][-1]=1
#                 double_data.append(data[i])
#                 double_data.append(data[j])
#             if data[j][3]=='occurSince' and data[i][3]=='occurUntil':
#                 data[i][-1]=1
#                 data[j][-1]=1
#                 double_data.append(data[j])
#                 double_data.append(data[i])
#-----------------------------------

#print(len(double_data))
# with open("./wiki/wiki.pkl", "wb") as file:
#     pickle.dump(double_data, file)
with open("./wiki/wiki.pkl", "rb") as file:
    double_data = pickle.load(file)

# print(len(data))
print(len(double_data))
#print(double_data)
# for i in double_data:
#     print(i)
#3把成对数据放到一个数据上
single_data=[]
#temp_list=[]
#定义时间匹配模式
#time_re=r"\"(.*?)-"
for i in range(0,len(double_data),2):
    temp_list=[]
    time1=double_data[i][4]
    time2=double_data[i+1][4]
    #sequence=re.findall(sequence_re,double_data[i])
    temp_list.append(double_data[i][0])
    temp_list.append(double_data[i][1])
    temp_list.append(double_data[i][2])
    temp_list.append(time1)
    temp_list.append(time2)
    single_data.append(temp_list)
#文件保存
# with open("./wiki/wiki_single.pkl", "wb") as file:
#     pickle.dump(single_data, file)
#df = pd.DataFrame(single_data)
#df.to_excel('./yago15k/yago15k_train.xlsx', index=False)
#时间跨度 1513-2017，平均
timelist=[]
for s in single_data:
    timelist.append(int(s[3]))
    timelist.append(int(s[4]))
    if int(s[3]) <30 or int (s[4])< 30:
        print('hhh')
time1513to1515=[]
#endtime=1912
total_count=0#总数量
intime_count=0#未过期数量
time_gap=5#时间间隔

for time in range(1980,2021,time_gap):
    count=0
    outtime_count=0
    total_count=0
    time_gap_count=0
    #noguoqi=0
    #endtime=1955
    fivefinal=[]
    #tempdata=[]
    for data in single_data:
        tempdata = []

        if int(data[3])<=time or int(data[4])<=time:#排除未来的信息
            total_count=total_count+1
            #print(count)
            if int(data[4])<time-time_gap:#所有过期的知识
                outtime_count=outtime_count+1
            if int(data[4])<time-time_gap and int(data[4])>=time-time_gap-time_gap:## 近五年内过期的知识
                time_gap_count=time_gap_count+1
                tempdata=data.copy()
                tempdata.append(0)
                fivefinal.append(tempdata)
                #data.pop()
            if int(data[4])>=time-time_gap:#统计所有不过期的知识
                #noguoqi=noguoqi+1
                tempdata=data.copy()

                tempdata.append(1)
                fivefinal.append(tempdata)
                #data.pop()

    intime=total_count-outtime_count
    #print(outtime_count)
    print("year:{}-{},total_num:{},unexpired_information:{},time_gap:{},time_gap_count:{}".format(time-time_gap,time,total_count,intime,time_gap,time_gap_count))
    #print(noguoqi)
    #print(fivefinal)
    print(len(fivefinal))
    with open("./wiki/wiki"+str(time-time_gap)+"_"+str(time)+"_5"+".pkl","wb") as file:
        pickle.dump(fivefinal, file)