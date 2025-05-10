import requests
from bs4 import BeautifulSoup
import pickle
import os
#os.system('echo ^G')
with open("./wiki/wiki_single.pkl", "rb") as file:
    single_data = pickle.load(file)
Q_total=[]
for s in single_data:
    if 'Q' in s[0]:
        Q_total.append(s[0])
    if 'Q' in s[2]:
        Q_total.append(s[2])
#print(Q_total)
#提取所有Q后set一下
Q_total=list(set(Q_total))
sum=0
Q_information=[]
Q_No_information=[]
Q_fail=[]
#Q_total:8458
#--------------------------------------------------------
begin=7800
end=8000
#end2=8458
#----------------------------------------------------------
for i in range(begin,end):
    Q=Q_total[i]
    #Q630491
    # 目标网页 URL
    #print(Q,sum,sum/len(Q_total))
    #Q=Q_total[189]
    sum=sum+1
    url = "https://www.wikidata.org/wiki/"+Q
    proxies = {
        "http": "http://127.0.0.1:7890",  # Clash 的 HTTP 代理端口
        "https": "http://127.0.0.1:7890"  # Clash 的 HTTPS 代理端口
    }
    # 设置请求头，模拟浏览器访问
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    # 发送 HTTP 请求获取网页内容
    response = requests.get(url, headers=headers,proxies=proxies)
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取标签
        #label_text = soup.find_all('span', class_='wikibase-labelview-text')[-1].text

        #label = soup.find('meta', attrs={'property': 'og:title'})['content']
        #print("label 内容:", label)  # 输出：Marius Lăcătuș

        # 提取 description 的内容
        if soup.find('meta', attrs={'name':'description'}) != None and soup.find('meta', attrs={'property': 'og:title'})!=None:
            label = soup.find('meta', attrs={'property': 'og:title'})['content']
            description = soup.find('meta', attrs={'name':'description'})['content']
            print("label 内容:", label)  # 输出：Marius Lăcătuș
            print("description 内容:", description)  # 输出：Romanian footballer
            Q_information.append([Q,label,description])
        elif soup.find('meta', attrs={'property': 'og:title'})!=None:
            label = soup.find('meta', attrs={'property': 'og:title'})['content']
            description=label
            print('only label')
            Q_information.append([Q, label, description])
        elif soup.find('meta', attrs={'name':'description'}) != None:
            description = soup.find('meta', attrs={'name': 'description'})['content']
            label=description
            print('only description')
            Q_information.append([Q, label, description])
        else:
            print("no information")
            Q_No_information.append(Q)
    else:
        print("网页请求失败，状态码:", response.status_code)
        Q_fail.append(Q)

with open("./wiki/Q_information"+str(begin)+"_"+str(end)+'.pkl',"wb") as file:
    pickle.dump(Q_information, file)
with open("./wiki/Q_No_information"+str(begin)+"_"+str(end)+'.pkl',"wb") as file:
    pickle.dump(Q_No_information, file)
with open("./wiki/Q_fail"+str(begin)+"_"+str(end)+'.pkl',"wb") as file:
    pickle.dump(Q_fail, file)
#os.system('echo ^G')
print(len(Q_No_information)+len(Q_fail))