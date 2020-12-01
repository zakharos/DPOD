"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
import scipy
import scipy.ndimage


def convert_to_homogenous(points):
    """
    Convert to homogenous coordinates
    Args:
        points: Input points

    Returns: Homogenized points

    """
    if points.shape[0] != 4:
        temp = np.ones((4, points.shape[1]))
        temp[:3, :] = points
        points = temp

    return points


def transform_points(points, transformation):
    """
    Apply transformation to the object points
    Args:
        points: Input points (non-homogeneous)
        transformation: Transformation matrix to apply

    Returns: transformed points

    """
    points = convert_to_homogenous(points)

    return np.dot(transformation, points)


def measure_add(model, gt_transformation, predicted_transformation):
    """
    Measure ADD metric
    Args:
        model: 3D object model
        gt_transformation: Ground truth transformation
        predicted_transformation: Predicted transformation

    Returns: ADD score

    """
    vertices_gt = transform_points(model.vertices.T, gt_transformation).T
    vertices_predicted = transform_points(model.vertices.T, predicted_transformation).T

    diff = vertices_gt - vertices_predicted
    diff = (diff * diff).sum(axis=-1)
    diff = np.mean(np.sqrt(diff))

    return diff / model.diameter_points


def measure_add_symmetric(model, gt_transformation, predicted_transformation):
    """
    Measure ADD metric for symmetric objects
    Args:
        model: 3D object model
        gt_transformation: Ground truth transformation
        predicted_transformation: Predicted transformation

    Returns: ADD score

    """
    vertices_gt = transform_points(model.vertices.T, gt_transformation).T
    vertices_predicted = transform_points(model.vertices.T, predicted_transformation).T

    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(vertices_gt)

    distances, _ = neighbors.kneighbors(vertices_predicted)

    diff = np.mean(distances)

    return diff / model.diameter_points


def measure_reprojection_error(model, camera, gt_transformation, predicted_transformation):
    """
    Measure reprojection error
    Args:
        model: 3D object model
        camera: Camera matrix
        gt_transformation: Ground truth transformation
        predicted_transformation: Predicted transformation

    Returns: Reprojection error value

    """
    proj_2d_gt = project_points(model.vertices.T, gt_transformation, camera).T
    proj_2d_pred = project_points(model.vertices.T, predicted_transformation, camera).T

    norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=1)
    pixel_dist = np.mean(norm)

    return pixel_dist


# Source: https://github.com/tensorflow/models/blob/master/research/object_detection/core/box_list_ops.py
def measure_iou(gtbbox, boxlist):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      gtbbox: BoxList holding N boxes
      boxlist: BoxList holding M boxes

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = measure_bbox_intersection(gtbbox, boxlist)
    areas = measure_bbox_area(gtbbox)
    areas2 = measure_bbox_area(boxlist)
    unions = (
            np.expand_dims(areas, 1) + np.expand_dims(areas2, 0) - intersections)

    return np.where(
        np.equal(intersections, 0.0),
        np.zeros_like(intersections), intersections / unions)


# Source: https://github.com/tensorflow/models/blob/master/research/object_detection/core/box_list_ops.py
def measure_bbox_intersection(gtbbox, boxlist2):
    """
    Compute pairwise intersection areas between boxes.

    Args:
      gtbbox: BoxList holding ! boxes
      boxlist2: BoxList holding M boxes

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """

    y_min1, x_min1, y_max1, x_max1 = np.split(gtbbox, axis=1, indices_or_sections=4)
    y_min2, x_min2, y_max2, x_max2 = np.split(boxlist2, indices_or_sections=4, axis=1)
    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

    return intersect_heights * intersect_widths


def measure_bbox_area(boxlist):
    """
    Computes area of boxes.

    Args:
      boxlist: BoxList holding N boxes

    Returns: A tensor with shape [N] representing box areas.

    """
    y_min, x_min, y_max, x_max = np.split(boxlist, indices_or_sections=4, axis=1)
    return np.squeeze((y_max - y_min) * (x_max - x_min), 1)


def init_3d_bbox(model):
    """
    Estimate the 3D bounding box given the 3D object model
    Args:
        model: 3D object model

    Returns: estimated 3D bounding box

    """
    minx, maxx = model.minx, model.maxx
    miny, maxy = model.miny, model.maxy
    minz, maxz = model.minz, model.maxz

    bb = []
    bb.append([minx, miny, minz, 1])
    bb.append([minx, maxy, minz, 1])
    bb.append([minx, miny, maxz, 1])
    bb.append([minx, maxy, maxz, 1])
    bb.append([maxx, miny, minz, 1])
    bb.append([maxx, maxy, minz, 1])
    bb.append([maxx, miny, maxz, 1])
    bb.append([maxx, maxy, maxz, 1])
    bb.append([0, 0, 0, 1])

    return np.array(bb, dtype=np.float).T


def project_points(points, transformation, intrinsic_mat):
    """
    Project point given the transformation and camera matrix
    Args:
        points: Object points
        transformation: Pose transformation
        intrinsic_mat: Camera matrix

    Returns: Transformed points

    """
    points = convert_to_homogenous(points)

    transformed_points = transform_points(points, np.dot(intrinsic_mat, transformation))
    transformed_points /= transformed_points[-1, :]

    return transformed_points[:-1]


def predict_pose_uv(cam, uv_region, model, return_inliers=False):
    """
    Predict pose given UV correspondences
    Args:
        cam: Camera matrix
        uv_region: UV region
        model: Object model
        return_inliers: Bool to return inliers

    Returns: Estimated pose

    """
    nonzero_mask = uv_region[:, :, 0] > 0

    uv_values = uv_region[nonzero_mask]

    uv_region_0 = uv_values[:, 0]
    uv_region_1 = uv_values[:, 1]

    points_3d = model.uv_to_3d[uv_region_0, uv_region_1]
    points_3d_mask = model.uv_to_3d_filled[uv_region_0, uv_region_1][:, 0]
    points_3d = points_3d[points_3d_mask]

    grid_row, grid_column = np.nonzero((nonzero_mask).astype(np.int64))

    grid_row = grid_row[points_3d_mask]
    grid_column = grid_column[points_3d_mask]
    image_points = np.empty((len(grid_row), 2))
    image_points[:, 0] = grid_row
    image_points[:, 1] = grid_column

    object_points = points_3d

    if return_inliers:
        predicted_pose, n_inliers = solvePnP(cam, image_points, object_points, return_inliers)
        predicted_pose = predicted_pose[:3]
        return predicted_pose, n_inliers
    else:
        predicted_pose = solvePnP(cam, image_points, object_points, return_inliers)
        predicted_pose = predicted_pose[:3]
        return predicted_pose


def predict_pose_uvw(cam, uvw_region, model, return_inliers=False):
    """
    Predict pose given UVW correspondences
    Args:
        cam: camera matrix
        uvw_region: UVW region
        model: object model
        return_inliers: bool to return inliers

    Returns: estimated pose

    """
    nonzero_mask = uvw_region[:, :, 0] > 0
    uvw_values = uvw_region[nonzero_mask]

    uvw_region_u = uvw_values[:, 0] * (model.maxx - model.minx) + model.minx
    uvw_region_v = uvw_values[:, 1] * (model.maxy - model.miny) + model.miny
    uvw_region_w = uvw_values[:, 2] * (model.maxz - model.minz) + model.minz
    points_3d = np.stack([uvw_region_u, uvw_region_v, uvw_region_w], axis=1)

    grid_row, grid_column = np.nonzero(nonzero_mask.astype(np.int64))

    image_points = np.empty((len(grid_row), 2))
    image_points[:, 0] = grid_row
    image_points[:, 1] = grid_column

    object_points = points_3d

    if return_inliers:
       predicted_pose, n_inliers = solvePnP(cam, image_points, object_points, return_inliers)
       predicted_pose = predicted_pose[:3]
       return predicted_pose, n_inliers
    else:
       predicted_pose = solvePnP(cam, image_points, object_points, return_inliers)
       predicted_pose = predicted_pose[:3]
       return predicted_pose


def init_predicted_bb(height_preprocessed, region_mask, width_preprocessed):
    """
    Initialize predicted bounding box
    Args:
        height_preprocessed: Preprocessed bounding box height
        region_mask: Mask of the region of interest
        width_preprocessed: Preprocessed bounding box width

    Returns: Predicted bounding box

    """
    grid_row, grid_column = np.nonzero(region_mask)

    row_top, row_bottom = grid_row.min() / height_preprocessed, grid_row.max() / height_preprocessed
    column_left, column_right = grid_column.min() / width_preprocessed, grid_column.max() / width_preprocessed

    predicted_bb = np.array([[row_top, column_left, row_bottom, column_right]])

    return predicted_bb


def solvePnP(cam, image_points, object_points, return_inliers=False, ransac_iter=250):
    """
    Solve PnP problem using resulting correspondences
    Args:
        cam: Camera matrix
        image_points: Correspondence points on the image
        object_points: Correspondence points on the model
        return_inliers: Bool for inliers return
        ransac_iter: Number of RANSAC iterations

    Returns: Resulting object pose (+ number of inliers)

    """
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    if image_points.shape[0] < 4:
        pose = np.eye(4)
        inliers = []
    else:
        image_points[:, [0, 1]] = image_points[:, [1, 0]]
        object_points = np.expand_dims(object_points, 1)
        image_points = np.expand_dims(image_points, 1)

        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points.astype(float), cam,
                                                                           dist_coeffs, iterationsCount=ransac_iter,
                                                                          reprojectionError=1.)[:4]

        # Get a rotation matrix
        pose = np.eye(4)
        if success:
            pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            pose[:3, 3] = np.squeeze(translation_vector)

        if inliers is None:
            inliers = []

    if return_inliers:
        return pose, len(inliers)
    else:
        return pose


def get_regions(mask_id):
    """
    Get detected interest regions
    Args:
        mask_id: Estimated mask

    Returns: Resulting region integer ndarray and the number of regions
    """
    # Post-processing
    mask = (mask_id != 0).astype(int)
    mask_separated = mask

    # detect separated regions
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    mask_regions, num_regions = scipy.ndimage.label(mask_separated, structure=s)

    return mask_regions, num_regions
