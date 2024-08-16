# -*- coding:utf-8 -*-
'''
Author: wudengyang
Date: 2022-12-27 10:41
Description: 
    compare 2 video by frames
'''
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='compare video')
parser.add_argument('--video_path', default='', type=str, help='video')
parser.add_argument('--video_path1', default='', type=str, help='video')
args = parser.parse_args()

video_path = args.video_path
video_path1 = args.video_path1
cap = cv2.VideoCapture(video_path)
cap1 =cv2.VideoCapture(video_path1)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")
if (cap1.isOpened()== False): 
    print("Error opening video stream or file")
    
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Frame1', cv2.WINDOW_NORMAL)
#import pdb; pdb.set_trace()
while(cap.isOpened()|cap1.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret, frame1 = cap1.read()
    
    if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        cv2.imshow('Frame1', frame1)

    # Press Q on keyboard to  exit
    key = cv2.waitKey(0)
    if key==27:
        exit()

cap.release()
cap1.release()

cv2.destroyAllWindows()
