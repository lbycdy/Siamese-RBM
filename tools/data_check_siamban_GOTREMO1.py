import json
import os

anno = '/home/lbycdy/cdh_json_new_230524/catdoghorse_lee_muin_new_newlabel230524.json'
data = json.load(open(anno))
path = r'/home/lbycdy/cdh_json_new_230524/data.txt'

file = open(path,'w+')
for key1 in data:
    file.write("{")
    a = data[key1]
    for i in range(len(a)):
        anno1 = []
        final = ''
        for key2 in a[i]:
                    # print(anno1)
            if key2 == "image_path":
                root = a[i][key2]
                root1 = root[36:]

                # print(root)
                # final = os.path.join(key1,root)
                # final = root
                # print(final)
                # file.write('"'+final+'":'+str(anno1)+',')
                # file.write(final)
                # file.write('":')
                # file.write(str(anno1))
                # file.write(',')
            if key2 == "boxes":
                b = a[i][key2]

                for j in range(len(b)):
                    if  b[j]['cid'] != 0:
                        anno2 = [(b[j]['xmin']),(b[j]['ymin']),(b[j]['xmax']),(b[j]['ymax']),b[j]['cid']]
                        anno1.append(anno2)

            if key2 == "root":
                # root = a[i][key2]
                c = a[i][key2]
                d = c.split('/')
                e = d[-1]
                # file.write('"' + root1 + '":' + str(anno1) + ',')

                # e = root[37:]
                # print(c)
                # print(e)
                final = os.path.join(e,root)
                print(final)

                file.write('"' + root + '":' + str(anno1) + ',')