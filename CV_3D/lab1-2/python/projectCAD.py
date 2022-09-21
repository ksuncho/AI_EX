from scipy.io import loadmat
from utils import estimate_params, estimate_pose, project
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

# load image, cad, x, X
S = loadmat('../data/PnP.mat')
image = S['image']
cad = S['cad']
x = S['x']
X = S['X']

# calculate camera matrix
P = estimate_pose(x, X)
K, R, t = estimate_params(P)

# 1. plot x, xProj on the image

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

xProj = project(P, X)
ax1.imshow(image)
ax1.scatter(x[0, :], x[1, :], c='r', marker='.')
ax1.scatter(xProj[0, :], xProj[1,:], c='b', marker='o',alpha=0.5)

# 2. plot 3d mesh from CAD data.
cad_vis = o3d.geometry.TriangleMesh()
cad_vis.vertices = o3d.utility.Vector3dVector(cad['vertices'][0][0])
cad_vis.triangles = o3d.utility.Vector3iVector(cad['faces'][0][0] - 1)
cad_vis.compute_vertex_normals()
o3d.visualization.draw_geometries([cad_vis])

# 3. plot projected CAD vertrices onto the image
cad_x = project(P, np.transpose(cad['vertices'][0][0]))
ax2.imshow(image)
ax2.scatter(cad_x[0,:], cad_x[1, :], c='r', s=0.2)
plt.show()