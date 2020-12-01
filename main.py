"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Sergey Zakharov
"""

import sys
import argparse
import configparser

from pipelines.train import train
from pipelines.test import test as test
from networks.resnet_uvw import resnet18 as dpod_uvw
from networks.resnet_uv import resnet18 as dpod_uv
import utils.data as data


def main():
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='config.ini', help='config file')
    parser.add_argument('--test', '-t', action='store_true', help='test network')

    # Parse arguments
    args = parser.parse_args()

    # Read config file
    cfg = args.config
    cfgparser = configparser.ConfigParser()
    res = cfgparser.read(cfg)
    if len(res) == 0:
        print("Error: None of the config files could be read")
        sys.exit(1)

    # Correspondence type
    corr_type = data.read_cfg_string(cfgparser, 'input', 'corr_type', default='')
    if corr_type == 'uv':
        dpod = dpod_uv
    elif corr_type == 'uvw':
        dpod = dpod_uvw

    # Execution
    if not args.test:
        train(cfgparser, test_function=test, create_cnn=dpod)
    else:
        test(cfgparser, create_cnn=dpod)


if __name__ == '__main__':
    main()
