import open3d as o3d
import numpy as np
import trimesh
import cv2
from matplotlib import pyplot as plt

"""
1) make cube with triangle mesh
"""

# mesh data consists of vertices and triangles
# Fill the appropriate values in the vertices array and triangles array to make a zero-centered cube
# with length 2.

cube = o3d.geometry.TriangleMesh()
###############################################################
##########################  TODO  #############################
###############################################################
vertices = np.array([[-1., -1., -1.],
                     [-1., -1., 1.],
                     [-1., 1., -1],
                     [-1., 1., 1.],
                     [1., -1., -1.],
                     [1., -1., 1.],
                     [1., 1., -1.],
                     [1., 1., 1.]])

triangles = np.array([[4, 7, 5],
                      [4, 6, 7],
                      [0, 2, 4],
                      [2, 6, 4],
                      [0, 1, 2],
                      [1, 3, 2],
                      [1, 5, 7],
                      [1, 7, 3],
                      [2, 3, 7],
                      [2, 7, 6],
                      [0, 4, 1],
                      [1, 4, 5]], dtype=np.int32)
###############################################################
###############################################################
cube.vertices = o3d.utility.Vector3dVector(vertices)
cube.triangles = o3d.utility.Vector3iVector(triangles)
cube = cube.compute_vertex_normals()
o3d.visualization.draw_geometries([cube])

"""
2) Add texture to a mesh
"""
text = cv2.imread('../data/cube_texture.png')
text = cv2.cvtColor(text, cv2.COLOR_BGR2RGB)

###############################################################
##########################  TODO  #############################
###############################################################
v_uv = np.array([[0.5, 0.66],
                 [0.75, 0.33],
                 [0.5, 0.66],
                 [0.5, 0.66],
                 [0.75, 0.66],
                 [0.75, 0.33],
                 [0.25, 0.66],
                 [0.25, 1],
                 [0.5, 0.66],
                 [0.25, 1],
                 [0.5, 1],
                 [0.5, 0.66],
                 [0.25, 0.66],
                 [0.25, 0.33],
                 [0, 0.66],
                 [0.25, 0.33],
                 [0, 0.33],
                 [0, 0.66],
                 [0.25, 0.33],
                 [0.5, 0.33],
                 [0.5, 0],
                 [0.25, 0.33],
                 [0.5, 0],
                 [0.25, 0],
                 [1, 0.66],
                 [1, 0.33],
                 [0.75, 0.33],
                 [1, 0.66],
                 [0.75, 0.33],
                 [0.75, 0.66],
                 [0.25, 0.66],
                 [0.5, 0.66],
                 [0.25, 0.33],
                 [0.25, 0.33],
                 [0.5, 0.66],
                 [0.5, 0.33]])
###############################################################
###############################################################

cube.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
cube.textures = [o3d.geometry.Image(text)]
o3d.io.write_triangle_mesh('../data/cube_texture.obj', cube)
