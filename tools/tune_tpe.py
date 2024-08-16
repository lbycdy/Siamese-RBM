from __future__ import absolute_import
import os
import argparse
import numpy as np

from siamban.utils.model_load import load_pretrain
from toolkit.utils.region import vot_overlap
# from test_ocean import auc_otb, eao_vot
from easydict import EasyDict as edict
from toolkit.datasets import VOTDataset
from toolkit.evaluation import EAOBenchmark, OPEBenchmark
from siamban.models.model_builder import ModelBuilder
from siamban.core.config import cfg
from siamban.tracker.siamban_tracker import SiamBANTracker
from os.path import join, realpath, dirname, exists
import ray
import cv2
from siamban.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect, poly_iou

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.suggest import HyperOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
from pprint import pprint
import json
import glob

parser = argparse.ArgumentParser(description='parameters for Ocean tracker')
parser.add_argument('--arch', dest='arch', default='Ocean',
                    help='architecture of model')
parser.add_argument('--resume', default='/home/insur/work/siamban-master/experiments/siamban_r50_l234/model.pth', type=str, required=True,
                    help='resumed model')
parser.add_argument('--config', default='/home/ispur/work/siamban-acm/experiments/siamban_r50_l234/siamban_acm_nobox.yaml', type=str, help='config file')
parser.add_argument('--cache_dir', default='./TPE_results', type=str, help='directory to store cache')
parser.add_argument('--gpu_nums', default=8, type=int, help='gpu numbers')
parser.add_argument('--trial_per_gpu', default=4, type=int, help='trail per gpu')
parser.add_argument('--dataset', default='VOT2018', type=str, help='dataset')
parser.add_argument('--align', default='True', type=str, help='align')
parser.add_argument('--online', default=False, type=bool, help='online flag')
parser.add_argument("--gpu_id", default="0", type=str, help="gpu id")

args = parser.parse_args()

print('==> However TPE is slower than GENE')

# prepare tracker
info = edict()
info.arch = args.arch
info.dataset = args.dataset
info.epoch_test = False
if args.online:
    info.align = False
else:
    info.align = True if 'VOT' in args.dataset and args.align == 'True' else False
info.online = args.online
info.TRT = 'TRT' in args.arch
if info.TRT:
    info.align = False

args.resume = os.path.abspath(args.resume)

def load_dataset(dataset):
    """
    support OTB and VOT now
    TODO: add other datasets
    """
    info = {}

    if 'OTB' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]
            info[v]['name'] = v

    elif 'VOT' in dataset and (not 'VOT2019RGBT' in dataset) and (not 'VOT2020' in dataset):
        base_path = join('/home/inspur/work/siamban-acm/tools/', '../testing_dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'VOT2020' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = open(gt_path, 'r').readlines()
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'RGBT234' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['infrared_imgs'] = [join(base_path, path_name, 'infrared', im_f) for im_f in
                                        info[v]['infrared_imgs']]
            info[v]['visiable_imgs'] = [join(base_path, path_name, 'visible', im_f) for im_f in
                                        info[v]['visiable_imgs']]
            info[v]['infrared_gt'] = np.array(info[v]['infrared_gt'])  # 0-index
            info[v]['visiable_gt'] = np.array(info[v]['visiable_gt'])  # 0-index
            info[v]['name'] = v

    elif 'VOT2019RGBT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            in_image_path = join(video_path, 'ir', '*.jpg')
            rgb_image_path = join(video_path, 'color', '*.jpg')
            in_image_files = sorted(glob.glob(in_image_path))
            rgb_image_files = sorted(glob.glob(rgb_image_path))

            assert len(in_image_files) > 0, 'please check RGBT-VOT dataloader'
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'infrared_imgs': in_image_files, 'visiable_imgs': rgb_image_files, 'gt': gt, 'name': video}
    elif 'VISDRONEVAL' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'annotations')
        attr_path = join(base_path, 'attributes')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'VISDRONETEST' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'initialization')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',').reshape(1, 4)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'GOT10KVAL' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.remove('list.txt')
        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'GOT10K' in dataset:  # GOT10K TEST
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.remove('list.txt')
        for video in videos:
            video_path = join(seq_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': [gt], 'name': video}

    elif 'LASOT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', dataset)
        json_path = join(realpath(dirname(__file__)), '../../dataset', dataset + '.json')
        jsons = json.load(open(json_path, 'r'))
        testingvideos = list(jsons.keys())

        father_videos = sorted(os.listdir(base_path))
        for f_video in father_videos:
            f_video_path = join(base_path, f_video)
            son_videos = sorted(os.listdir(f_video_path))
            for s_video in son_videos:
                if s_video not in testingvideos:  # 280 testing videos
                    continue

                s_video_path = join(f_video_path, s_video)
                # ground truth
                gt_path = join(s_video_path, 'groundtruth.txt')
                gt = np.loadtxt(gt_path, delimiter=',')
                gt = gt - [1, 1, 0, 0]
                # get img file
                img_path = join(s_video_path, 'img', '*jpg')
                image_files = sorted(glob.glob(img_path))

                info[s_video] = {'image_files': image_files, 'gt': gt, 'name': s_video}
    elif 'DAVIS' in dataset and 'TEST' not in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', 'DAVIS')
        list_path = join(realpath(dirname(__file__)), '../../dataset', 'DAVIS', 'ImageSets', dataset[-4:],
                         'val.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            info[video] = {}
            info[video]['anno_files'] = sorted(glob.glob(join(base_path, 'Annotations/480p', video, '*.png')))
            info[video]['image_files'] = sorted(glob.glob(join(base_path, 'JPEGImages/480p', video, '*.jpg')))
            info[video]['name'] = video
    elif 'YTBVOS' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../dataset', 'YTBVOS', 'valid')
        json_path = join(realpath(dirname(__file__)), '../../dataset', 'YTBVOS', 'valid', 'meta.json')
        meta = json.load(open(json_path, 'r'))
        meta = meta['videos']
        info = dict()
        for v in meta.keys():
            objects = meta[v]['objects']
            frames = []
            anno_frames = []
            info[v] = dict()
            for obj in objects:
                frames += objects[obj]['frames']
                anno_frames += [objects[obj]['frames'][0]]
            frames = sorted(np.unique(frames))
            info[v]['anno_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in frames]
            info[v]['anno_init_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in anno_frames]
            info[v]['image_files'] = [join(base_path, 'JPEGImages', v, im_f + '.jpg') for im_f in frames]
            info[v]['name'] = v

            info[v]['start_frame'] = dict()
            info[v]['end_frame'] = dict()
            for obj in objects:
                start_file = objects[obj]['frames'][0]
                end_file = objects[obj]['frames'][-1]
                info[v]['start_frame'][obj] = frames.index(start_file)
                info[v]['end_frame'][obj] = frames.index(end_file)

    else:
        raise ValueError("Dataset not support now, edit for other dataset youself...")

    return info

def track_tune(tracker, net, video, config):
    arch = config['arch']
    benchmark_name = config['benchmark']
    resume = config['resume']
    hp = config['hp']  # scale_step, scale_penalty, scale_lr, window_influence

    tracker_path = join('test', (benchmark_name + resume.split('/')[-1].split('.')[0] +
                                     '_penalty_k_{:.4f}'.format(hp['penalty_k']) +
                                     '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                     '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .
    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in benchmark_name:
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = join(video_path, video['name'] + '_001.txt')
    elif 'GOT10K' in benchmark_name:
        re_video_path = os.path.join(tracker_path, video['name'])
        if not exists(re_video_path): os.makedirs(re_video_path)
        result_path = os.path.join(re_video_path, '{:s}.txt'.format(video['name']))
    else:
        result_path = join(tracker_path, '{:s}.txt'.format(video['name']))

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        if benchmark_name.startswith('OTB'):
            return tracker_path
        elif benchmark_name.startswith('VOT') or benchmark_name.startswith('GOT10K'):
            return 0
        else:
            print('benchmark not supported now')
            return

    start_frame, lost_times, toc = 0, 0, 0

    regions = []  # result and states[1 init / 2 lost / 0 skip]


    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            # target_pos = np.array([cx, cy])
            # target_sz = np.array([w, h])
            gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            state = tracker.init(im, gt_bbox_)  # init tracker
            # location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(im, hp=hp)  # track
            # location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            pred_bbox = state['bbox']
            overlap = vot_overlap(pred_bbox, gt[f], (im.shape[1], im.shape[0]))
            # b_overlap = poly_iou(gt[f], location) if 'VOT' in benchmark_name else 1
            if overlap > 0:
                regions.append(pred_bbox)
            else:
                regions.append([float(2)])
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append([float(0)])

    # save results for OTB
    if 'OTB' in benchmark_name or 'LASOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VISDRONE' in benchmark_name  or 'GOT10K' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                p_bbox = x.copy()
                fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
    elif 'VOT' in benchmark_name:
        with open(result_path, "w") as fin:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    if 'OTB' in benchmark_name or 'VIS' in benchmark_name or 'VOT' in benchmark_name or 'GOT10K' in benchmark_name:
        return tracker_path
    else:
        print('benchmark not supported now')

# def auc_otb(tracker, net, config):
#     """
#     get AUC for OTB benchmark
#     """
#     dataset = load_dataset(config['benchmark'])
#     video_keys = list(dataset.keys()).copy()
#     random.shuffle(video_keys)
#
#     for video in video_keys:
#         result_path = track_tune(tracker, net, dataset[video], config)
#
#     auc = eval_auc_tune(result_path, config['benchmark'])
#
#     return auc

def eao_vot(tracker, net, config):
    dataset = load_dataset(config['benchmark'])
    video_keys = sorted(list(dataset.keys()).copy())

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    re_path = result_path.split('/')[0]
    tracker = result_path.split('/')[-1]

    # debug
    print('======> debug: results_path')
    print(result_path)
    print(os.system("ls"))
    print(join('/home/inspur/work/siamban-acm/tools/', '../testing_dataset', args.dataset))

    # give abs path to json path
    data_path = join('/home/inspur/work/siamban-acm/tools/', '../testing_dataset', args.dataset)
    dataset = VOTDataset(config['benchmark'], data_path)

    dataset.set_tracker(re_path, tracker)
    benchmark = EAOBenchmark(dataset)
    eao = benchmark.eval(tracker)
    eao = eao[tracker]['all']

    return eao


# fitness function
def fitness(config, reporter):
    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.resume).cuda().eval()
    # rebuild tracker
    tracker = SiamBANTracker(model)

    print('pretrained model has been loaded')
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if 'Ocean' in args.arch:
        penalty_k = config["penalty_k"]
        scale_lr = config["scale_lr"]
        window_influence = config["window_influence"]
        # small_sz = config["small_sz"]
        # big_sz = config["big_sz"]
        # ratio = config["ratio"]

        model_config = dict()
        model_config['benchmark'] = args.dataset
        model_config['arch'] = args.arch
        model_config['resume'] = args.resume
        model_config['hp'] = dict()
        model_config['hp']['penalty_k'] = penalty_k
        model_config['hp']['window_influence'] = window_influence
        model_config['hp']['lr'] = scale_lr
        # model_config['hp']['small_sz'] = small_sz
        # model_config['hp']['big_sz'] = big_sz
        # model_config['hp']['ratio'] = ratio
    else:
        raise ValueError('not supported model')

    # VOT and Ocean
    if args.dataset.startswith('VOT'):
        eao = eao_vot(tracker, model, model_config)
        print(
            "penalty_k: {0}, scale_lr: {1}, window_influence: {2}, eao: {3}".format(
                penalty_k, scale_lr, window_influence, eao))
        reporter(EAO=eao)

    # # OTB and Ocean
    # if args.dataset.startswith('OTB'):
    #     auc = auc_otb(tracker, model, model_config)
    #     print(
    #         "penalty_k: {0}, scale_lr: {1}, window_influence: {2}, small_sz: {3}, big_sz: {4}, ratio: {6}, eao: {5}".format(
    #             penalty_k, scale_lr, window_influence, small_sz, big_sz, auc.item(), ratio))
    #     reporter(AUC=auc)


if __name__ == "__main__":
    # the resources you computer have, object_store_memory is shm
    # ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8,  object_store_memory=50000000000)
    ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8, object_store_memory=500000000)
    tune.register_trainable("fitness", fitness)

    # load config
    cfg.merge_from_file(args.config)
    if 'Ocean' in args.arch:
        params = {
            "penalty_k": hp.quniform('penalty_k', 0.0, 1.0, 0.001),
            "scale_lr": hp.quniform('scale_lr', 0.0, 1.0, 0.001),
            "window_influence": hp.quniform('window_influence', 0.0, 1.0, 0.001),
            # "small_sz": hp.choice("small_sz", [255]),
            # "big_sz": hp.choice("big_sz", [287, 303, 319]),
            # "ratio": hp.quniform('ratio', 0.7, 1, 0.01),
        }
    if 'VOT' not in args.dataset or not args.align:
        params['ratio'] = hp.choice("ratio", [1])

    print('tuning range: ')
    pprint(params)

    tune_spec = {
        "zp_tune": {
            "run": "fitness",
            "resources_per_trial": {
                "cpu": 1,  # single task cpu num
                "gpu": 1.0 / args.trial_per_gpu,  # single task gpu num
            },
            "num_samples": 2000,  # sample hyperparameters times
            "local_dir": args.cache_dir
        }
    }

    # stop condition for VOT and OTB
    if args.dataset.startswith('VOT'):
        stop = {
            "EAO": 0.46,  # if EAO >= 0.6, this procedures will stop
            # "timesteps_total": 100, # iteration times
        }
        tune_spec['zp_tune']['stop'] = stop

        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            metric='EAO',
            mode='max',
            max_t=400,
            grace_period=20
        )
        # max_concurrent: the max running task
        # algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*args.trial_per_gpu + 1, reward_attr="EAO")
        algo = HyperOptSearch(params, max_concurrent=args.gpu_nums * args.trial_per_gpu + 1, metric='EAO', mode='max')

    elif args.dataset.startswith('OTB') or args.dataset.startswith('VIS') or args.dataset.startswith('GOT10K'):
        stop = {
            # "timesteps_total": 100, # iteration times
            "AUC": 0.80
        }
        tune_spec['zp_tune']['stop'] = stop
        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            reward_attr="AUC",
            max_t=400,
            grace_period=20
        )
        algo = HyperOptSearch(params, max_concurrent=args.gpu_nums * 2 + 1, reward_attr="AUC")  #
    else:
        raise ValueError("not support other dataset now")

    tune.run_experiments(tune_spec, search_alg=algo, scheduler=scheduler)


