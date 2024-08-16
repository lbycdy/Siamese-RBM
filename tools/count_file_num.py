import os
dir_count = 0
file_count = 0
file_path = '/home/lbycdy/datasets/OCean/VID/crop511-anno'
for root, dirs, filenames in os.walk(file_path):
    for dir in dirs:
        dir_count += 1
    for file in filenames:
        file_count += 1
print('dir_count ', dir_count)
print('file_count ', file_count)