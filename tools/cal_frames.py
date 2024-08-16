# -*- coding:utf-8 -*-
'''
Author: wudengyang
Date: 2023-01-11 21:38
Description: 
    calculate frames of video from file folder, txt, json...
'''
import cv2
import os
import glob

def cal_frames_from_file_folder(root_dir):
    videos_path_list = glob.glob(root_dir + '/*.mp4', recursive=True) + \
                       glob.glob(root_dir + '/*.MP4', recursive=True) + \
                       glob.glob(root_dir + '/*.mkv', recursive=True) + \
                       glob.glob(root_dir + '/*.webm', recursive=True)
    frames = 0
    for video_path in videos_path_list:
        cap = cv2.VideoCapture(video_path)
        frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frames

def cal_frames_from_txt1(savefile_path):
    with open(savefile_path, 'r') as fp:
        meta_data = fp.read().splitlines()
    frames = 0
    for video_path in meta_data:
        cap = cv2.VideoCapture(video_path)
        frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frames

def cal_frames_from_txt2(txt_file):
    txt_file_list = os.listdir(txt_file)
    videos_info = []
    for txt_name in txt_file_list:
        txt_path = os.path.join(txt_file, txt_name)
        with open(txt_path, 'r') as fp:
            video_info = []
            for line in fp:
                info = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
                video_info.append([int(i) for i in info])
            videos_info.append(video_info)
    total_frames = 0
    for video_info in videos_info:
        for info in video_info:
            total_frames += info[1]-info[0]
    return total_frames, videos_info

def cal_frames_from_json(jsons_dir):
    for video_name in os.listdir(jsons_dir):
        json_path = os.path.join(jsons_dir, video_name)
        with open(json_path, 'r') as f1:
            video_info = json.load(f1)
        total_frame = 0
        for video in video_info:
            total_frame += len(video)
        print(f'{video_name}:{total_frame}')

def video_file(file_path):
    num1, num2, num3, num4, num5, num6, num7 = 0, 0, 0, 0, 0, 0, 0
    # 循环文件夹下的文件并取出文件的名称(含后缀名)
    for video_name in os.listdir(file_path):
        video_path = os.path.join(file_path, video_name)
        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames > 0 and frames <= 10000:
            num1 += 1
        elif frames > 10000 and frames <= 20000:
            num2 += 1
        elif frames > 20000 and frames <= 30000:
            num3 += 1
        elif frames > 30000 and frames <= 40000:
            num4 += 1
        elif frames > 40000 and frames <= 50000:
            num5 += 1
        elif frames > 50000 and frames <= 60000:
            num6 += 1
        elif frames > 60000:
            num7 += 1
    return num1, num2, num3, num4, num5, num6, num7





if __name__ == '__main__':
    root_dir = '/mnt/data/GOT/GOT10K/got10k/crop511/train'
    frames = cal_frames_from_file_folder(root_dir)
    print(f'total num of frames: {frames}')

    # savefile_path = '/home/wudengyang/work/Scripts/sp.txt'
    # frames = cal_frames_from_txt1(savefile_path)
    # print(f'total num of frames: {frames}')

    # jsons_dir = '/mnt/data/GOT/GOT10K/got10k'
    # cal_images_num_from_json(jsons_dir)
    #
    # num1, num2, num3, num4, num5, num6, num7 = video_file("/home/wudengyang/StreetDanceVideo/video_clip")
    # print(f'0~1万:{num1}, 1~2万:{num2}, 2~3万:{num3}, 3~4万:{num4}, 4~5万:{num5}, 5~6万:{num6}, >6万:{num7}')
    # print(num1+num2+num3+num4+num5+num6+num7)
    #
    # total_frames, videos_info = cal_frames_from_txt2('/home/wudengyang/StreetDanceVideo_Rename202301txt')
    # print(f'total_frames:{total_frames}')  # 6505367
    #
    # root_dir = '/home/wudengyang/StreetDanceVideo/video_clip'
    # total_frames = cal_frames_from_file_folder(root_dir)
    # print(f'total_frames:{total_frames}')  # 127475