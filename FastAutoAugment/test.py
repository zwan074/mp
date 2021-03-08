import pathlib
import sys
import numpy as np
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import itertools
import json
import logging
import math
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from FastAutoAugment.common import get_logger, EMA, add_filehandler
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.lr_scheduler import adjust_learning_rate_resnet
from FastAutoAugment.metrics import accuracy, Accumulator, CrossEntropyLabelSmooth
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.tf_port.rmsprop import RMSpropTF
from FastAutoAugment.aug_mixup import CrossEntropyMixUpLabelSmooth, mixup
from warmup_scheduler import GradualWarmupScheduler

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='./data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='test.pth')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation-interval', type=int, default=5)
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()

    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'], args.dataroot, args.cv_ratio, split_idx=args.cv, multinode=(args.local_rank >= 0))
    data = iter(trainloader).next() 
    print(len ( data[0]) )
    print( data[0][0].size() )