import re
import csv
import pandas as pd
import pickle

train_path="../mmkb-master/TemporalKGs/yago15k/yago15k_train.txt"
test_path="../mmkb-master/TemporalKGs/yago15k/yago15k_test.txt"
valid_path="../mmkb-master/TemporalKGs/yago15k/yago15k_valid.txt"
data=[]
num=0
#1删除无时间数据
with open(train_path, 'r', encoding='utf-8') as file:
    for line in file:
        #print(line.strip())  # 使用 strip() 去掉每行末尾的换行符
        #data.append(line)
        if '##' in line:
            #print(line)
            data.append(line)
with open(test_path, 'r', encoding='utf-8') as file:
    for line in file:
        #print(line.strip())  # 使用 strip() 去掉每行末尾的换行符
        #data.append(line)
        if '##' in line:
            #print(line)
            data.append(line)
with open(valid_path, 'r', encoding='utf-8') as file:
    for line in file:
        #print(line.strip())  # 使用 strip() 去掉每行末尾的换行符
        #data.append(line)
        if '##' in line:
            #print(line)
            data.append(line)
#2，使数据成对
#定义正则表达式,匹配尖括号
sequence_re=r"<(.*?)>"
double_data=[]
for i in range(0,len(data)-1):
    sequence1=re.findall(sequence_re,data[i])
    sequence2=re.findall(sequence_re,data[i+1])

    if "Since" in data[i] and "Until" and sequence1[0]==sequence2[0] and sequence1[1]==sequence2[1] and sequence1[2]==sequence2[2]:
        #i=i+2
        double_data.append(data[i])
        double_data.append(data[i+1])


# print(len(data))
# print(len(double_data))
#print(double_data)
# for i in double_data:
#     print(i)
#3把成对数据放到一个数据上
single_data=[]
#temp_list=[]
#定义时间匹配模式
time_re=r"\"(.*?)-"
for i in range(0,len(double_data),2):
    temp_list=[]
    time1=re.findall(time_re,double_data[i])
    time2=re.findall(time_re,double_data[i+1])
    sequence=re.findall(sequence_re,double_data[i])
    temp_list.append(sequence[0])
    temp_list.append(sequence[1])
    temp_list.append(sequence[2])
    temp_list.append(time1[0])
    temp_list.append(time2[0])
    single_data.append(temp_list)
#文件保存
# with open("./yago15k/yago15k_train.csv","w",newline="",encoding="utf-8") as file:
#     writer=csv.writer(file)
#     writer.writerow(single_data)
#df = pd.DataFrame(single_data)
#df.to_excel('./yago15k/yago15k_train.xlsx', index=False)
#时间跨度 1513-2017，平均
#timelist

time1513to1515=[]
endtime=1912
total_count=0#总数量
intime_count=0#未过期数量
time_gap=5#时间间隔

for time in range(1997,2018,time_gap):
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
    with open("./yago15k/yago"+str(time-time_gap)+"_"+str(time)+"_5"+".pkl","wb") as file:
        pickle.dump(fivefinal, file)