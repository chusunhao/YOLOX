#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp

# python tools/train.py -f exps/example/spark/yolox_s.py -d 1 -b 8 --fp16 -o -c yolox_s.pth
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "saprk_" + os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/SPARK"
        self.train_ann = "train_labels.json"
        self.val_ann = "validate_labels.json"

        self.num_classes = 11

        self.max_epoch = 40
        self.data_num_workers = 4
        self.eval_interval = 1
        self.warmup_epochs = 1
        self.no_aug_epochs = 10
