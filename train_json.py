import json
c = "/home/lbycdy/datasets/OCean/GOT10K/train_withclassname.json"
data = json.load(open(c))
for video in data:
    for track in data[video]:
        print(data[video][track]['cls'])