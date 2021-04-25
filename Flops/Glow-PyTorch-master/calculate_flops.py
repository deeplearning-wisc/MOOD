import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from datasets import get_CIFAR10, get_SVHN
from model import Glow


if 1:
    model = Glow(
        image_shape=(32,32,3),
        hidden_channels=512,
        K=32,
        L=3,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="affine",
        LU_decomposed=True,
        y_classes=10,
        learn_top=True,
        y_condition=False,
    )
    print(model)
    model = model.cuda()

    import pickle
    data_input = open('x.pkl','rb')
    image = pickle.load(data_input)
    data_input.close()
    
    pre = model(image[0:1], None)