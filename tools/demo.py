from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--save', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()
# args.config = '/home/ethan/siamban-acm/experiments/siamban_r50_l234/20220919MobilenetV2_BANACM_4GPU126_OneMidFeatSigmoid_Changepadding.yaml'
# args.snapshot = '/home/ethan/siamban-acm/experiments/siamban_r50_l234/checkpoint_e20.pth'

args.config = '/home/ethan/work/siamban-acm/experiments/siamban_r50_l234/20220930MobilenetV2_BANACM_3GPU126_OneMidFeatSigmoid_Changepadding_syncBN_rankloss.yaml'
args.snapshot = '/media/ethan/OldDisk/home/ethan/Models/Results/0016_GOT/20220930MobilenetV2_BANACM_3GPU126_OneMidFeatSigmoid_Changepadding_syncBN_rankloss/checkpoint_e20.pth'


args.config = '/home/ethan/work/siamban-acm/experiments/siamban_r50_l234/20221017MobilenetV2_BANACM_3GPU126_ThreeFeatSigmoid_Changepadding_rankloss.yaml'
args.snapshot = '/media/ethan/OldDisk/home/ethan/Models/Results/0016_GOT/20221017MobilenetV2_BANACM_3GPU126_ThreeFeatSigmoid_Changepadding_rankloss/checkpoint_e20.pth'

args.video_name = "/home/ethan/siamban-acm/demo/bag.avi"
args.video_name = '/media/ethan/OldDisk/home/ethan/test_videos/Hor_remain/d1.mp4'
args.video_name = "/home/ethan/work/siamban-acm/tools/20220926_GOT_facenouse.mp4" # (767, 151, 231, 312)
args.video_name = "/home/ethan/20221101FaceGOT.avi" #
args.video_name = "/home/ethan/20221101FaceGOT_2.avi"
args.video_name = "/home/ethan/Downloads/normalvideo_got.mp4"


args.video_name = 0
def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

flagwriteorgvideo = False
def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    # model.eval().to(device)

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    if flagwriteorgvideo:
        out = cv2.VideoWriter("20220926_GOT_facenouse.mp4", fourcc, 10, (1920, 1080))
    for frame in get_frames(args.video_name):

        if first_frame:
            # build video writer
            if args.save:
                if args.video_name.endswith('avi') or \
                    args.video_name.endswith('mp4') or \
                    args.video_name.endswith('mov'):
                    cap = cv2.VideoCapture(args.video_name)
                    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
                else:
                    fps = 30

                save_video_path = args.video_name.split(video_name)[0] + video_name + '_tracking.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_size = (frame.shape[1], frame.shape[0]) # (w, h)
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size)
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            if sum(init_rect) != 0:
                print("init_rect:",init_rect)
                tracker.init(frame, init_rect)
                first_frame = False
                if flagwriteorgvideo:
                    out.write(frame)
        else:
            if flagwriteorgvideo:
                out.write(frame)

            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                s = outputs["best_score"]
                cx = int(bbox[0]+bbox[2]/2)
                cy = int(bbox[1]+bbox[3]/2)
                cv2.putText(frame, "%f" % s, (int(cx), int(cy)), cv2.FONT_HERSHEY_COMPLEX, fontScale=1,color=(0, 0, 255), thickness=1)
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            key = cv2.waitKey(1)
            if key==27:
                exit()

        if args.save:
            video_writer.write(frame)
    
    if args.save:
        video_writer.release()


if __name__ == '__main__':
    main()
