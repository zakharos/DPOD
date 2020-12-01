"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Sergey Zakharov
"""

import os
import torch

from utils import data
from data.train import PatchDatasetRGB2CORR


def train(cfg, test_function, create_cnn):
    """
    Training function
    Args:
        cfg: Config file
        test_function: Evaluation function
        create_cnn: Neural network class
    """

    # Setup device
    device_name = data.read_cfg_string(cfg, 'optimization', 'device', default='cpu')
    if device_name == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Set model and optimizer
    restore_path = data.read_cfg_string(cfg, 'input', 'restore_net', default='')
    dpod = create_cnn(pretrained=True).to(device)

    if len(restore_path) > 0:
        dpod.load_state_dict(torch.load(restore_path), strict=False)
        print('Loaded model')

    # Logs
    log_dir = data.read_cfg_string(cfg, 'test', 'dir', default='log')
    os.makedirs(log_dir, exist_ok=True)

    # Prepare the data
    batch_size = data.read_cfg_int(cfg, 'train', 'batch_size', default=32)
    cpu_threads = data.read_cfg_int(cfg, 'optimization', 'cpu_threads', default=3)
    trainset = PatchDatasetRGB2CORR(cfg)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_threads)

    # Run the optimizer
    epochs = data.read_cfg_int(cfg, 'train', 'epochs', default=1000)
    for epoch in range(epochs):

        # Training loop
        for i, (rgb, corr, mask) in enumerate(trainloader):
            loss_corr, loss_mask = dpod.optimize(rgb.to(device),
                                                 corr.to(device),
                                                 mask.to(device))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses: - Corr: {:.6f}, - Mask: {:.6f}'.format(
                epoch, i * len(rgb), len(trainloader.dataset),
                       100. * i / len(trainloader), loss_corr.item(), loss_mask.item()))

        # Save networks, analyze performance
        if epoch != 0 and epoch % data.read_cfg_int(cfg, 'train', 'analyze_epoch', default=100) == 0:

            # Store net
            net_dir = os.path.join(log_dir, 'net')
            os.makedirs(net_dir, exist_ok=True)
            torch.save(dpod.state_dict(), os.path.join(net_dir, 'model_{}.pt'.format(epoch)))
            print('Saved network')

            # Test performance
            test_function(cfg, cnn=dpod)
