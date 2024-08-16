import json
path = '/home/lbycdy/data.txt'


with open(path,'r',encoding='utf-8')as f:                     #打开txt文件
    data = f.readlines()
    json_data = json.dumps(data)
    with open('data.json','w') as f:
        f.write(json_data)
