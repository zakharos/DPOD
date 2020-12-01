"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT
"""

import numpy as np
from torchvision import transforms
import yaml
from yaml import CLoader


def read_cfg_string(cfg, section, key, default):
    """
    Read string from a config file
    Args:
        cfg: config file
        section: [section] of the config file
        key: key to be read
        default: value if couldn't be read

    Returns: resulting string

    """
    if cfg.has_option(section, key):
        return cfg.get(section, key)
    else:
        return default


def read_cfg_int(cfg, section, key, default):
    """
    Read int from a config file
    Args:
        cfg: config file
        section: [section] of the config file
        key: key to be read
        default: value if couldn't be read

    Returns: resulting int

    """
    if cfg.has_option(section, key):
        return cfg.getint(section, key)
    else:
        return default


def read_cfg_float(cfg, section, key, default):
    """
    Read float from a config file
    Args:
        cfg: config file
        section: [section] of the config file
        key: key to be read
        default: value if couldn't be read

    Returns: resulting float

    """
    if cfg.has_option(section, key):
        return cfg.getfloat(section, key)
    else:
        return default


def read_cfg_bool(cfg, section, key, default):
    """
    Read bool from a config file
    Args:
        cfg: config file
        section: [section] of the config file
        key: key to be read
        default: value if couldn't be read

    Returns: resulting bool

    """
    if cfg.has_option(section, key):
        return cfg.get(section, key) in ['True', 'true']
    else:
        return default


def read_cfg_cam(cfg, section, key, default):
    """
    Read camera matrix from a config file
    Args:
        cfg: config file
        section: [section] of the config file
        key: key to be read
        default: value if couldn't be read

    Returns: resulting camera matrix

    """
    if cfg.has_option(section, key):
        str = cfg.get(section, key).split(',')
        cam = np.array([[float(str[0]), 0., float(str[1])],
                        [0., float(str[2]), float(str[3])],
                        [0., 0., 1.]])
        return cam
    else:
        return default


def read_cfg_intrinsics(cfg, section, key, default):
    """
    Read intrinsics from a config file
    Args:
        cfg: config file
        section: [section] of the config file
        key: key to be read
        default: value if couldn't be read

    Returns: resulting intrinsic dict

    """
    if cfg.has_option(section, key):
        str = cfg.get(section, key).split(',')

        intinsics = {
            "x0": 320,
            "y0": 240,
            "fx": float(str[0]),
            "fy": float(str[2])
        }
        return intinsics
    else:
        return default


def read_cfg_resolution(cfg, section, key, default):
    """
    Read input image resolution from a config file
    Args:
        cfg: config file
        section: [section] of the config file
        key: key to be read
        default: value if couldn't be read

    Returns: resulting resolution tuple

    """
    if cfg.has_option(section, key):
        str = cfg.get(section, key).split(',')
        resolution = (int(str[0]), int(str[1]))
        return resolution
    else:
        return default


def load_yaml(dir_path):
    """
    Load YAML file to read datasets
    Args:
        dir_path: path to the file

    Returns: corresponding Python object

    """
    with open(dir_path, 'r') as stream:
        data = yaml.load(stream, Loader=CLoader)

    return data


# Data transformation and augmentation routines
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

preprocess_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

preprocess_rgb = transforms.Compose([
    transforms.ColorJitter(brightness=32. / 255., contrast=0.5, saturation=0.5, hue=0.2), #TODO was 0.05
    transforms.ToTensor(),
    normalize
])

preprocess_rgb_refinement = transforms.Compose([
    transforms.ColorJitter(brightness=1. / 255., contrast=0.00001, saturation=0.0001, hue=0.05), #TODO was 0.05
    transforms.ToTensor(),
    normalize
])