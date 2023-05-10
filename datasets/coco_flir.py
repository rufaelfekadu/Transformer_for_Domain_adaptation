#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/17 15:00
# @Author  : Hao Luo
# @File    : msmt17.py

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class CocoFlir(BaseImageDataset):
    """
    Coco Flir dataset
    """
    dataset_dir = ''

    def __init__(self, source_data='../dataset/sgada_data/mscoco/train/mscoco_train.txt', target_data='../dataset/sgada_data/flir/train/flir_train.txt', target_test= "../dataset/sgada_data/flir/val/flir_val.txt", pid_begin=0, verbose=True, **kwargs):
        super(CocoFlir, self).__init__()
        # root_train = root_train
        # root_valid = root_val

        self.train_source_dir = osp.dirname(source_data)
        self.train_target_dir = osp.dirname(target_data)
        self.valid_dataset_dir = osp.dirname(target_test)

        self.train_name = osp.basename(source_data).split('.')[0]
        self.valid_name = osp.basename(target_test).split('.')[0]

        self.pid_begin = pid_begin
        source = self._process_dir(source_data, self.train_source_dir)
        target = self._process_dir(target_data, self.train_target_dir)
        test = self._process_dir(target_test, self.valid_dataset_dir)

        
        if verbose:
            print("=> Coco-Flir loaded")
            self.print_dataset_statistics(source, target, test)
            
        self.train = source
        self.valid = target
        self.test = test

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)   
        self.num_valid_pids, self.num_valid_imgs, self.num_valid_cams, self.num_valid_vids = self.get_imagedata_info(self.valid)
        self.num_test_pids = self.num_valid_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def print_dataset_statistics(self, train, valid, test):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_valid_pids, num_valid_imgs, num_valid_cams, num_targe_views = self.get_imagedata_info(valid)
        num_test_pids, num_test_imgs, num_test_cams, num_test_views = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("train {} and valid is {}".format(self.train_name, self.valid_name))
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  source   | {:5d} | {:8d} | {:9d}".format( num_train_pids, num_train_imgs, num_train_cams))
        print("  target   | {:5d} | {:8d} | {:9d}".format(num_valid_pids, num_valid_imgs, num_valid_cams))
        print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_cams))
        print("  ----------------------------------------")
        
    def _process_dir(self, list_path, dir_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin +pid, 0, 0, img_idx))
            pid_container.add(pid)
#             cam_container.add(camid)
#         print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset
    

if __name__=='__main__':

    source_dir = '../dataset/sgada_data/mscoco/train/mscoco_train.txt'
    target_dir = '../dataset/sgada_data/flir/train/flir_train.txt'
    test_dir = "../dataset/sgada_data/flir/val/flir_val.txt"

    dataset = CocoFlir(source_dir, target_dir, test_dir)
    print(dataset.train_source_dir)