import os
filePath = '/home/lbycdy/datasets/OCean/GOT10K/crop511/train'    #获得当前文件夹的路径
dir_count = 0
file_count = 0
for root,dirs,files in os.walk(filePath):    #遍历统计
    for dir in dirs:
        dir_count += 1 # 统计文件夹下的文件夹总个数
    for _ in files:
        file_count += 1   # 统计文件夹下的文件总个数
print ('dir_count ', dir_count) # 输出结果
print ('file_count ', file_count)              