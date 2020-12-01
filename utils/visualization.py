"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT
"""

import cv2


def bboxes_draw_on_img_bb(img, classes, bboxes, bbox_iou, thickness=2, color=(255, 255, 255)):
    """
    Draw 2D bounding boxes
    Args:
        img: Input image
        classes: Object classes
        bboxes: Bounding box coordinates
        bbox_iou: Bounding box iou
        thickness: Line thickness
        color: Line color
    """
    shape = img.shape

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]

        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))

        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

        # Draw text...
        if bbox_iou is not None:
            s = '%s/%.3f' % (classes[i], bbox_iou[0][i])
            p1 = (p1[0] - 5, p1[1])
            cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)


def bboxes_3d_draw_on_img_bb(img, vertices_list, thickness=2, color=(255, 255, 255)):
    """
    Draw 3D bounding boxes
    Args:
        img: Input image
        vertices_list: List of vertices
        thickness: Line thickness
        color: Line color
    """
    surfaces = [(2, 0, 1, 3), (4, 5, 7, 6), (2, 6, 7, 3), (0, 4, 5, 1)]
    for bb_ind, vertices in enumerate(vertices_list):
        for i, surface in enumerate(surfaces):

            points = [(int(round(vertices[vertex_ind, 0])),
                       int(round(vertices[vertex_ind, 1]))) for
                      vertex_ind in surface]

            cv2.line(img, points[0], points[1], color, thickness)
            cv2.line(img, points[1], points[2], color, thickness)
            cv2.line(img, points[2], points[3], color, thickness)
            cv2.line(img, points[3], points[0], color, thickness)

