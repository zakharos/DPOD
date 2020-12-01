"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT
"""


import plyfile
import numpy as np
import scipy
import scipy.spatial


class Model:
    def __init__(self):
        self.vertices = None
        self.indices = None
        self.colors = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.diameter = None
        self.frames = []

    def load(self, model):
        ply = plyfile.PlyData.read(model)
        self.vertices = np.zeros((ply['vertex'].count, 3))
        self.vertices[:, 0] = np.array(ply['vertex']['x'])
        self.vertices[:, 1] = np.array(ply['vertex']['y'])
        self.vertices[:, 2] = np.array(ply['vertex']['z'])

        self.bb = []
        self.minx, self.maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        self.miny, self.maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        self.minz, self.maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])
       
        self.bb.append([self.minx, self.miny, self.minz])
        self.bb.append([self.minx, self.maxy, self.minz])
        self.bb.append([self.minx, self.miny, self.maxz])
        self.bb.append([self.minx, self.maxy, self.maxz])
        self.bb.append([self.maxx, self.miny, self.minz])
        self.bb.append([self.maxx, self.maxy, self.minz])
        self.bb.append([self.maxx, self.miny, self.maxz])
        self.bb.append([self.maxx, self.maxy, self.maxz])
        self.bb = np.asarray(self.bb, dtype=np.float32)
        self.diameter = max(scipy.spatial.distance.pdist(self.bb, 'euclidean'))
        self.colors = np.zeros((ply['vertex'].count, 3))
        self.colors[:, 0] = np.array(ply['vertex']['blue'])
        self.colors[:, 1] = np.array(ply['vertex']['green'])
        self.colors[:, 2] = np.array(ply['vertex']['red'])
        self.indices = np.asarray(list(ply['face']['vertex_indices']))

        vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3)]
        self.collated = np.asarray(list(zip(self.vertices, self.colors)), vertices_type)

        self.uv_to_3d = np.empty((256, 256, 3))
        self.uv_to_3d_filled = np.zeros((256, 256, 1), dtype=np.bool)

        for i, color in enumerate(self.colors):
            u, v = int(color[0]), int(color[1])
            self.uv_to_3d[u, v] = self.vertices[i]
            self.uv_to_3d_filled[u, v] = True