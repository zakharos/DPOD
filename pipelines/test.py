"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Sergey Zakharov
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd

from utils import data, visualization
import data.test as dbs
import utils.evaluation as eval


def test(cfg, cnn=None, create_cnn=None, output_images=True, output_stats=True):
    """
    LineMOD evaluation function. Computes the necessary metrics.

    Args:
        cfg: Config file
        cnn: Neural network object
        create_cnn: Neural network class if no net object is given
        output_images: bool to save images

    Returns: Recall, Precision, F1, ADD 10/30/50

    """
    # Setup device
    from utils import data
    device_name = data.read_cfg_string(cfg, 'optimization', 'device', default='cpu')
    if device_name == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    cam = data.read_cfg_cam(cfg, 'test', 'intrinsics', default=None)

    # Set model and optimizer
    restore_path = data.read_cfg_string(cfg, 'input', 'restore_net', default='')

    if not cnn:
        dpod = create_cnn(pretrained=True).to(device)
        if len(restore_path) > 0:
            dpod.load_state_dict(torch.load(restore_path), strict=False)
    else:
        dpod = cnn
    dpod.eval()

    # Create log directories
    log_dir, mask_out_dir, path_out_2d, path_out_3d, path_out_poses, corr_out_dir = init_standard_dirs(cfg)

    # Initialize metric dictionaries
    model_to_recall, model_to_precision, model_to_f1 = {}, {}, {}
    model_to_add10, model_to_add30, model_to_add50 = {}, {}, {}

    # Prepare the data
    cpu_threads = data.read_cfg_int(cfg, 'optimization', 'cpu_threads', default=3)
    model_to_dataset = dbs.init_linemod_datasets(cfg)

    # Loop through models to test
    for model in model_to_dataset.keys():
        print('Processing model ' + model)

        # Load the data for a particular model
        testset = model_to_dataset[model]
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cpu_threads)

        n_detected_correctly = 0
        n_total_detections = 0

        all_distances = []
        count_add_10, count_add_30, count_add_50 = 0, 0, 0

        # Run the optimizer
        for i, (rgb_preprocessed, rgb_raw, (img_name,), gt_pose, gt_bbox) in enumerate(testloader):
            rgb_raw = rgb_raw.cpu().numpy()[0][:, :, ::-1]
            _, _, height_preprocessed, width_preprocessed = rgb_preprocessed.shape
            rgb_2d, rgb_3d = np.copy(rgb_raw), np.copy(rgb_raw)

            gt_pose = gt_pose.numpy()[0]
            gt_bbox = gt_bbox.numpy()[0]

            object_detected_correctly = False
            pose_estimated = False
            best_distance = 100.
            best_pose = None

            if i % 10 == 0:
                print('\t{}/{}'.format(i, len(testloader)))

            dpod.forward(rgb_preprocessed.to(device))

            # get net output
            mask_id, corr = dpod.read_network_output()
            mask_regions, num_regions = eval.get_regions(mask_id)

            big_enough_regions = 0

            pnp_bbs = []
            gt_bbs = []

            for region_ind in range(1, num_regions + 1, 1):
                region_mask = (mask_regions == region_ind).astype(int)
                region_area = np.count_nonzero(region_mask)

                if region_area > 32:
                    big_enough_regions += 1

                    predicted_bb = eval.init_predicted_bb(height_preprocessed, region_mask, width_preprocessed)
                    bbox_iou = float(eval.measure_iou(gt_bbox, predicted_bb)[0][0])

                    if bbox_iou > 0.5:
                        object_detected_correctly = True

                        # mask corr and mask_id
                        corr_region = corr * (np.transpose(np.tile(mask_regions, (corr.shape[2], 1, 1)),
                                                       (1, 2, 0)) == region_ind).astype(int)

                        corr_region = cv2.resize(corr_region.astype(np.uint8), None, fx=2., fy=2.,
                                               interpolation=cv2.INTER_LINEAR)

                        # get color <-> pixel correspondences (one to many)
                        pose_estimated = True
                        if dpod.type == 'uv':
                            predicted_pose = eval.predict_pose_uv(cam, corr_region, testset.model)
                        elif dpod.type == 'uvw':
                            predicted_pose = eval.predict_pose_uvw(cam, corr_region, testset.model)

                        if model in ("10", "11"):
                            distance = eval.measure_add_symmetric(testset.model, gt_pose, predicted_pose)
                        else:
                            distance = eval.measure_add(testset.model, gt_pose, predicted_pose)

                        if distance < best_distance:
                            best_distance = distance
                            best_pose = predicted_pose

            if pose_estimated:

                # Estimated bounding box
                bb3d = eval.init_3d_bbox(testset.model)
                transformed_bb = eval.project_points(bb3d, predicted_pose, cam)
                transformed_bb = np.transpose(transformed_bb, (1, 0))
                pnp_bbs.append(transformed_bb)
                all_distances.append(best_distance)

                # Ground truth bounding box
                gt_transformed_bb = eval.project_points(bb3d, gt_pose, cam)
                gt_transformed_bb = np.transpose(gt_transformed_bb, (1, 0))
                gt_bbs.append(gt_transformed_bb)

                # Evaluate ADD
                if best_distance <= 0.10:
                    count_add_10 += 1

                if best_distance <= 0.30:
                    count_add_30 += 1

                if best_distance <= 0.50:
                    count_add_50 += 1

            n_total_detections += big_enough_regions
            n_detected_correctly += int(object_detected_correctly)

            if output_images and pose_estimated:
                # Save mask
                cv2.imwrite(os.path.join(mask_out_dir, img_name + '_mask.png'),
                            mask_id * 255)
                # Save correspondences
                if dpod.type == 'uv':
                    cv2.imwrite(os.path.join(corr_out_dir, img_name + '_corr.png'),
                                np.concatenate((corr, np.zeros_like(corr[:, :, :1])), axis=2))
                elif dpod.type == 'uvw':
                    cv2.imwrite(os.path.join(corr_out_dir, img_name + '_corr.png'), corr)
                # Draw 2D bounding box
                visualization.bboxes_draw_on_img_bb(rgb_2d, None, predicted_bb, None)
                cv2.imwrite(os.path.join(path_out_2d, img_name) + '_bb.png', rgb_2d)

                # Draw 3D bounding box
                visualization.bboxes_3d_draw_on_img_bb(rgb_3d, np.array(gt_bbs),
                                                       color=(0, 255, 0))  # Ground truth
                visualization.bboxes_3d_draw_on_img_bb(rgb_3d, np.array(pnp_bbs),
                                                       color=(255, 0, 0))  # Prediction
                cv2.imwrite(os.path.join(path_out_3d, img_name) + '_bb.png', rgb_3d)

                # Save pose to TXT
                np.savetxt(path_out_poses + img_name + '.txt', best_pose)

        # Compute Recall, Precision, and F1
        recall = n_detected_correctly / len(testloader)

        if n_total_detections > 0:
            precision = n_detected_correctly / n_total_detections
        else:
            precision = 0

        if n_detected_correctly > 0:
            add_10 = count_add_10 / n_detected_correctly
            add_30 = count_add_30 / n_detected_correctly
            add_50 = count_add_50 / n_detected_correctly
        else:
            add_10 = 0.
            add_30 = 0.
            add_50 = 0.

        if recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        print('Recall: ', recall)
        print('Precision: ', precision)
        print('F1 ', f1)

        mean_distance = np.mean(all_distances)

        print('\ndistance: ', mean_distance)
        print('ADD 10: {:.6f}, ADD 30: {:.6f}, ADD 50: {:.6f}'.format(add_10, add_30, add_50))
        print('N correctly detected: ', n_detected_correctly)

        model_to_precision[model] = precision
        model_to_recall[model] = recall
        model_to_f1[model] = f1
        model_to_add10[model] = add_10
        model_to_add30[model] = add_30
        model_to_add50[model] = add_50

        # Store statistics
        if output_stats:
            data = {model: [precision, recall, f1, add_10, add_30, add_50]}
            df = pd.DataFrame.from_dict(data, orient='index', columns=['Precision', 'Recall', 'F1',
                                                                       'ADD 10%', 'ADD 30%', 'ADD 50%'])
            df.to_csv(log_dir + 'stats_{}.csv'.format(model))

        return model_to_recall, model_to_precision, model_to_f1, model_to_add10, model_to_add30, model_to_add50


def init_standard_dirs(cfg):
    """
    Initialize directories for statistics
    Args:
        cfg: Config file

    Returns: Paths to created directories

    """
    log_dir = data.read_cfg_string(cfg, 'test', 'dir', default='log')
    corr_out_dir = os.path.join(log_dir, 'corr/')
    mask_out_dir = os.path.join(log_dir, 'masks/')
    path_out_2d = os.path.join(log_dir, '2d/')
    path_out_3d = os.path.join(log_dir, '3d/')
    path_out_poses = os.path.join(log_dir, 'poses/')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(corr_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(path_out_2d, exist_ok=True)
    os.makedirs(path_out_3d, exist_ok=True)
    os.makedirs(path_out_poses, exist_ok=True)

    return log_dir, mask_out_dir, path_out_2d, path_out_3d, path_out_poses, corr_out_dir


