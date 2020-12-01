"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Sergey Zakharov
"""

import os
import numpy as np
from torch.utils.data.dataset import Dataset
import cv2
import glob
import torchvision.transforms.functional as F
from collections import defaultdict

from utils import data
from data.model import Model


class LinemodDataset(Dataset):
    def __init__(self, cfg, model):
        self.path = data.read_cfg_string(cfg, 'test', 'path_test_data', default=None)
        path_models = data.read_cfg_string(cfg, 'test', 'path_models', default=None)
        self.cam = data.read_cfg_cam(cfg, 'test', 'intrinsics', default=np.identity(3)).astype(np.float32)

        # Read camera parameters from config
        self.scale = data.read_cfg_float(cfg, 'train', 'image_scale', default=.5)

        # Find min/max frame numbers
        files = os.listdir(os.path.join(self.path, model, 'rgb'))

        self.images = []
        self.img_names = []

        for file in files:
            img_path = os.path.join(self.path, model, 'rgb', file)

            if 'mask' in file or 'dpt' in file:
                continue

            self.images.append(img_path)
            img_name = file.replace('.png', '').replace('_color', '')
            self.img_names.append(img_name)

        self.gt = data.load_yaml(os.path.join(self.path, model, 'gt.yml'))
        self.color_vertex_global, self.model = _get_vertex_colors([model], path_models)

        model_properties = data.load_yaml(path_models + 'models_info.yml')[int(model)]
        self.model.diameter_points = float(model_properties['diameter'])
        self.model_name = model

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rgb_raw = cv2.imread(self.images[idx])
        b, g, r = rgb_raw[:, :, 0], rgb_raw[:, :, 1], rgb_raw[:, :, 2]
        rgb_raw = np.stack((r, g, b), axis=2)
        original_height, original_width, _ = rgb_raw.shape

        rgb_preprocessed = cv2.resize(rgb_raw, None, fx=self.scale, fy=self.scale,
                                      interpolation=cv2.INTER_NEAREST)

        # Transform to torch format
        rgb_preprocessed = data.preprocess_test(rgb_preprocessed).float()

        img_name = self.img_names[idx]
        for id, gt_f in enumerate(self.gt[int(img_name)]):
            if gt_f['obj_id'] == int(self.model_name):
                gt_img = gt_f
                break

        gt_pose = self._read_gt_pose(gt_img)
        gt_bbox = self._read_gt_bbox(gt_img, original_height, original_width)

        return rgb_preprocessed, rgb_raw, img_name, gt_pose, gt_bbox

    def _read_gt_bbox(self, gt_img, original_height, original_width):
        gt_bbox = gt_img['obj_bb']

        column_left = gt_bbox[0] / original_width
        column_right = column_left + gt_bbox[2] / original_width

        row_top = gt_bbox[1] / original_height
        row_bottom = row_top + gt_bbox[3] / original_height

        gt_bbox = np.array(((row_top, column_left, row_bottom, column_right),))

        return gt_bbox

    def _read_gt_pose(self, gt_img):
        rotation = np.array(gt_img['cam_R_m2c']).reshape((3, 3))
        translation = np.array(gt_img['cam_t_m2c']).reshape((3,))

        gt_pose = np.empty((3, 4))
        gt_pose[:3, :3] = rotation
        gt_pose[:3, 3] = translation

        return gt_pose


def _get_vertex_colors(models, path):
    color_vertex_global = {}
    model = Model()

    for idx, model_name in enumerate(models):
        model_path = os.path.join(path, 'obj_' + model_name) + '.ply'
        model.load(model_path)

        # Store color <-> vertex for a model (one to many)
        color_vertex_tmp = defaultdict(list)
        for id, color in enumerate(model.colors[:, :2]):
            color_vertex_tmp[tuple(color.astype(int))].append(model.vertices[id].tolist())

        # Convert to one to one
        color_vertex = {}
        for color, vertices in color_vertex_tmp.items():
            color_vertex[color] = vertices[0]

        # Store object correspondences to a global list
        color_vertex_global[model_name] = color_vertex

    return color_vertex_global, model


def init_linemod_datasets(cfg):
    """
    Initialize LineMOD datasets per object
    Args:
        cfg: config file

    Returns: dict of datasets

    """
    models = data.read_cfg_string(cfg, 'test', 'models_test', default=None).split(',')
    models_to_datasets = {}

    for model in models:
        testset = LinemodDataset(cfg, model)
        models_to_datasets[model] = testset

    return models_to_datasets


class SimpleDataset(Dataset):
    def __init__(self, cfg, scale=0.5):
        self.path_models = data.read_cfg_string(cfg, 'test', 'path_models', default=None)
        self.model_names = data.read_cfg_string(cfg, 'test', 'models_test', default=None).split(',')
        self.path = data.read_cfg_string(cfg, 'test', 'path_test_data', default=None)
        self.obj_dict = {}

        # Store image names and models
        for key in self.model_names:
            self.images = glob.glob(os.path.join(self.path, key, 'rgb') + os.path.sep + '*.png')
            obj = Model()
            obj.load(os.path.join(self.path_models, 'obj_' + key) + '.ply')
            self.obj_dict[key] = obj

        self.scale = scale

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rgb = cv2.imread(self.images[idx])

        # BGR -> RGB
        rgb = rgb[:, :, ::-1]

        # Transform to torch format
        rgb = F.to_pil_image(rgb)
        rgb = data.preprocess_test(rgb)

        obj_key = os.path.basename(os.path.split(os.path.split(self.images[idx])[0])[0])

        return rgb, obj_key
