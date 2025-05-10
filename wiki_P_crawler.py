import requests
from bs4 import BeautifulSoup
import pickle

with open("./wiki/wiki_single.pkl", "rb") as file:
    single_data = pickle.load(file)
P_total=[]
for s in single_data:
    if 'P' in s[1]:
        P_total.append(s[1])
P_total=list(set(P_total))
url = "https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all"
proxies = {
    "http": "http://127.0.0.1:7890",  # Clash 的 HTTP 代理端口
    "https": "http://127.0.0.1:7890"  # Clash 的 HTTPS 代理端口
}
# 设置请求头，模拟浏览器访问
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
response = requests.get(url, headers=headers,proxies=proxies)
P_information=[]
P_No_information=[]
for P in P_total:


    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到包含P6的<a>标签
    P_link = soup.find('a', href='/wiki/Property:'+P, title='Property:'+P)

    # 如果找到了P6标签，提取其下方的两个<td>标签内容
    if P_link:
        # 找到P6标签所在的行
        row = P_link.find_parent('tr')
        if row:
            # 提取P6下方的两个<td>标签内容
            cells = row.find_all('td')
            if len(cells) >= 3:  # 确保有足够的<td>标签
                label = cells[1].text.strip()
                description = cells[2].text.strip()
                print("Label:", label)
                print("Description:", description)
                P_information.append([P,label,description])
            else:
                print("Not enough <td> tags found")
                P_No_information.append(P)
        else:
            print("No parent <tr> found f")
            P_No_information.append(P)
    else:
        print("P6 link not found")
        P_No_information.append(P)
with open("./wiki/P_information","wb") as file:
    pickle.dump(P_information, file)
with open("./wiki/P_No_information","wb") as file:
    pickle.dump(P_No_information, file)