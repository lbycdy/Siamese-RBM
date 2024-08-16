# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch import multiprocessing as mp

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from siamban.utils.lr_scheduler import build_lr_scheduler
from siamban.utils.log_helper import init_log, print_speed, add_file_handler
from siamban.utils.distributed import average_reduce_world_size
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from siamban.utils.model_load import load_pretrain, restore_from
from siamban.utils.average_meter import AverageMeter
from siamban.utils.misc import describe, commit
# from siamban.models.model_builder import ModelBuilder
# from siamban.models.model_OHME import OhemModelBuilder
from siamban.models.model_DogCatHorseAsPersonV022PerTKMobileOneWeightAdd_ACMOutPointMask_FrV006HeadPadding import ModelBuilder
from siamban.datasets.dataset_remocatdoghorseAsPerTKV2_PointTarget_FrV2AddNfet import BANDataset
from siamban.datasets.dataset_remocatdoghorseAsPerTKV3_PointTargetFrV1VidImgRand import BANDataset as BANDatasetV3

from siamban.datasets.dataset_remocatdoghorseAsPerTKV05_PointTargetFrV3ModifyShift import BANDataset as BANDatasetV05


from siamban.core.config import cfg
from siamban.models.backbone.mobileone_stride import reparameterize_model
from siamban.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_train,reparameterize_model_allskipscale

import socket
"""
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH=/home/inspur/work/siamban-master
python ../../tools/train_PT10.py --cfg 20220920BANResNet50_uniform_2GPU126_pt1.10_syncBN.yaml
"""
logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='/home/lbycdy/work/siamban-cow/experiments/lb20230302DogCatHorseAsPersonV007PerTKMobileOneWeightAdd_DV05VIRandPNMod3SFT7Smin0.5Cov0.35In160NoRk_4GPU127.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
parser.add_argument('--distributed-port', type=str, default='12355')
parser.add_argument('--distributed-addr', type=str, default='localhost')
args = parser.parse_args()

cfg.merge_from_file(args.cfg)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ipstr = s.getsockname()[0]
ipaddress_int = int(ipstr.split('.')[-1])
if ipaddress_int == 122:
    modelsaveroot = "/home/zhangming/sdc/Models/Results/0016_GOT"
elif ipaddress_int == 110:
    modelsaveroot = "/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Models/Results/0016_GOT"
elif ipaddress_int == 117:
    modelsaveroot = "/home/zhangming/7c1f78ec-abec-4146-b3d9-2b4f91a445f4/Models/Results/0016_GOT"
elif ipaddress_int == 122:
    modelsaveroot = "/home/zhangming/sdc/Models/Results/0016_GOT"
elif ipaddress_int == 120:
    modelsaveroot = "/home/zhangming/sdd/Models/Results/0016_GOT"
elif ipaddress_int == 113:
    modelsaveroot = "/home/zhangming/f87ff691-1e63-43a2-834c-ca90e9356c5f/Models/Results/0016_GOT"
elif ipaddress_int == 111:
    modelsaveroot = "/home/zhangming/47557749-05d8-4626-b22f-b1c43c475c5c/Models/Results/0016_GOT"
elif ipaddress_int in [126,127,125,128]:
    modelsaveroot = "/home/inspur/Models/Results/0016_GOT"
modelname = os.path.splitext(args.cfg)[0]
cfg.TRAIN.LOG_DIR = os.path.join(modelsaveroot, modelname)
cfg.TRAIN.SNAPSHOT_DIR = os.path.join(modelsaveroot, modelname)


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        print("start training backbone")
        logger.info("start training backbone")
        for param in model.backbone.parameters():
            param.requires_grad = True
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler

def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)



class Trainer:
    def __init__(self, rank, world_size):
        self.seed_torch(args.seed)
        self.init_distributed(rank, world_size)
        self.init_writer()
        self.init_datasets()
        self.init_model()
        self.train()
        self.cleanup()

    def seed_torch(self,seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        logger.info('Initializing distributed')
        os.environ['MASTER_ADDR'] = args.distributed_addr
        os.environ['MASTER_PORT'] = args.distributed_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def init_datasets(self):
        logger.info("build train dataset")
        if cfg.DATASET.DATASETID == 0:
            train_dataset = BANDataset(world_size=self.world_size,batchsize=cfg.TRAIN.BATCH_SIZE)
        elif cfg.DATASET.DATASETID == 1:
            train_dataset = BANDatasetV3(world_size=self.world_size,batchsize=cfg.TRAIN.BATCH_SIZE)
        elif cfg.DATASET.DATASETID == 2:
            train_dataset = BANDatasetV05(world_size=self.world_size, batchsize=cfg.TRAIN.BATCH_SIZE)
        logger.info("build dataset done")

        train_sampler = None
        torch.cuda.set_device(self.rank)

        if self.world_size > 1:
            train_sampler = DistributedSampler(train_dataset)
        self.train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=True,
                                  sampler=train_sampler)
    def init_writer(self):
        if self.rank == 0:
            if not os.path.exists(cfg.TRAIN.LOG_DIR):
                os.makedirs(cfg.TRAIN.LOG_DIR)
            init_log('global', logging.INFO)
            if cfg.TRAIN.LOG_DIR:
                add_file_handler('global',
                                 os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                                 logging.INFO)

            logger.info("Version Information: \n{}\n".format(commit()))
            logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

            # create tensorboard writer
            if self.rank == 0 and cfg.TRAIN.LOG_DIR:
                self.tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
            else:
                self.tb_writer = None
    def init_model(self):
        # if cfg.TRAIN.OHEM:
        #     self.model = OhemModelBuilder().to(self.rank)
        # else:
        self.model = ModelBuilder(cfg).to(self.rank)

        # load pretrained backbone weights
        if cfg.BACKBONE.PRETRAINED:
            logger.info("load backbone from %s"%cfg.BACKBONE.PRETRAINED)
            if cfg.BACKBONE.PRETRAINED_Mode==0:
                pretrained = torch.load(cfg.BACKBONE.PRETRAINED, map_location=torch.device('cpu'))
                logger.info(self.model.backbone.load_state_dict(pretrained, strict=False))

                if cfg.BACKBONE.TYPE  in ["mobileones16outtwo","mobileones8s16outtwo"]:
                    if cfg.TRAIN.MODE_REPARAMETERIZE==0:
                        self.model.backbone = reparameterize_model_allskipscale(self.model.backbone).to(self.rank)
                    else:
                        self.model.backbone = reparameterize_model_train(self.model.backbone).to(self.rank)
                else:
                    self.model.backbone = reparameterize_model(self.model.backbone)
            elif cfg.BACKBONE.PRETRAINED_Mode==1:
                if cfg.BACKBONE.TYPE == ["mobileones16outtwo","mobileones8s16outtwo"]:
                    if cfg.TRAIN.MODE_REPARAMETERIZE == 0:
                        self.model.backbone = reparameterize_model_allskipscale(self.model.backbone).to(self.rank)
                    else:
                        self.model.backbone = reparameterize_model_train(self.model.backbone).to(self.rank)
                else:
                    self.model.backbone = reparameterize_model(self.model.backbone)
                pretrained = torch.load(cfg.BACKBONE.PRETRAINED, map_location=torch.device('cpu'))
                state_dict = self.model.backbone.state_dict()
                for key in state_dict:
                    keysrc = "backbone."+key
                    state_dict[key] = pretrained[keysrc]
                logger.info(self.model.backbone.load_state_dict(state_dict, strict=False))

        # build optimizer and lr_scheduler
        self.optimizer, self.lr_scheduler = build_opt_lr(self.model,
                                                             cfg.TRAIN.START_EPOCH)
        # resume training
        if cfg.TRAIN.RESUME:
            logger.info("resume from {}".format(cfg.TRAIN.RESUME))
            assert os.path.isfile(cfg.TRAIN.RESUME), \
                '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
            self.model, self.optimizer, cfg.TRAIN.START_EPOCH = \
                restore_from(self.model, self.optimizer, cfg.TRAIN.RESUME)
        # load pretrain
        elif cfg.TRAIN.PRETRAINED:
            load_pretrain(self.model, cfg.TRAIN.PRETRAINED)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
        logger.info(self.lr_scheduler)
        logger.info("model prepare done")
    def train(self):
        cur_lr = self.lr_scheduler.get_cur_lr()

        average_meter = AverageMeter()

        def is_valid_number(x):
            return not(math.isnan(x) or math.isinf(x) or x > 1e4)

        num_per_epoch = len(self.train_loader.dataset) // \
            cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * self.world_size)
        start_epoch = cfg.TRAIN.START_EPOCH
        epoch = start_epoch

        if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
                self.rank == 0:
            os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)
        logger.info("model\n{}".format(describe(self.model_ddp.module)))
        end = time.time()
        print(cfg.TRAIN.SNAPSHOT_DIR)

        for idx, data in enumerate(self.train_loader):
            #if idx % 100 == 0 and idx != 0:
            #    torch.save(
            #        {
            #            'state_dict': self.model_ddp.module.state_dict(),
            #        },
            #        cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint%d_e%d.pth' % (self.rank, epoch))
            if epoch != idx // num_per_epoch + start_epoch:
                epoch = idx // num_per_epoch + start_epoch

                if self.rank == 0:
                    torch.save(
                            {'epoch': epoch,
                             'state_dict': self.model_ddp.module.state_dict(),
                             'optimizer': self.optimizer.state_dict()},
                            cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))

                if epoch == cfg.TRAIN.EPOCH:
                    return

                if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                    logger.info('start training backbone.')
                    self.optimizer, self.lr_scheduler = build_opt_lr(self.model, epoch)
                    logger.info("model\n{}".format(describe(self.model_ddp.module)))

                self.lr_scheduler.step(epoch)
                cur_lr = self.lr_scheduler.get_cur_lr()
                logger.info('epoch: {}'.format(epoch+1))

            tb_idx = idx
            if idx % num_per_epoch == 0 and idx != 0:
                for idx, pg in enumerate(self.optimizer.param_groups):
                    logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                    if self.rank == 0:
                        self.tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                             pg['lr'], tb_idx)

            data_time = average_reduce_world_size(time.time() - end,self.world_size,self.rank)
            if self.rank == 0:
                self.tb_writer.add_scalar('time/data', data_time, tb_idx)

            outputs = self.model_ddp(data,self.rank)
            loss = outputs['total_loss']

            if is_valid_number(loss.data.item()):
                self.optimizer.zero_grad()
                loss.backward()
                if self.rank == 0 and cfg.TRAIN.LOG_GRADS:
                    log_grads(self.model_ddp.module, self.tb_writer, tb_idx)

                # clip gradient
                clip_grad_norm_(self.model_ddp.parameters(), cfg.TRAIN.GRAD_CLIP)
                self.optimizer.step()

            batch_time = time.time() - end
            batch_info = {}
            batch_info['batch_time'] = average_reduce_world_size(batch_time,self.world_size,self.rank)
            batch_info['data_time'] = average_reduce_world_size(data_time,self.world_size,self.rank)
            for k, v in sorted(outputs.items()):
                batch_info[k] = average_reduce_world_size(v.data.item(),self.world_size,self.rank)

            average_meter.update(**batch_info)

            if self.rank == 0:
                for k, v in batch_info.items():
                    self.tb_writer.add_scalar(k, v, tb_idx)

                if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                    info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                                epoch+1, (idx+1) % num_per_epoch,
                                num_per_epoch, cur_lr)
                    for cc, (k, v) in enumerate(batch_info.items()):
                        if cc % 2 == 0:
                            info += ("\t{:s}\t").format(
                                    getattr(average_meter, k))
                        else:
                            info += ("{:s}\n").format(
                                    getattr(average_meter, k))
                    logger.info(info)
                    print_speed(idx+1+start_epoch*num_per_epoch,
                                average_meter.batch_time.avg,
                                cfg.TRAIN.EPOCH * num_per_epoch)
            end = time.time()
            if self.rank == 0 and idx % 1000 == 0:
                torch.save(
                    {'epoch': epoch,
                     'state_dict': self.model_ddp.module.state_dict(),
                     'optimizer': self.optimizer.state_dict()},
                    cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (epoch))
        if self.rank == 0:
            torch.save(
                {'epoch': epoch,
                 'state_dict': self.model_ddp.module.state_dict(),
                 'optimizer': self.optimizer.state_dict()},
                cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (epoch))
    def cleanup(self):
        dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)
