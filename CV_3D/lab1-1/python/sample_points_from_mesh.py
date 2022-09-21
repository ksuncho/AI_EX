import trimesh
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""
In this example, we will convert a mesh into a point cloud.
Namely, you will sample points from the mesh.
"""
vertex_sample_num = 100
surface_sample_num = 1000
# read mesh data
sphere = trimesh.primitives.Sphere()
edges = sphere.edges_unique
length = sphere.edges_unique_length
g = nx.Graph()
for edge, L in zip(edges, length):
    g.add_edge(*edge, length=L)

# 1. Farthest Point Sampling

points = []

###############################################################
##########################  TODO  #############################
###############################################################

# 1) start with an arbitrary vertex.
points.append(0)

# 2) perform Dijkstra's algorithm.
for it in range(vertex_sample_num):
    path_length = nx.multi_source_dijkstra_path_length(g, points)
    nodes = np.fromiter(path_length.keys(), dtype=np.int32)
    lengths = np.fromiter(path_length.values(), dtype=np.int32)
# 3) add farthest node into points.
    points.append(nodes[np.argmax(lengths)])

###############################################################
###############################################################

sampled_points = sphere.vertices[points]

# visualize sampled points with matplotlib with matplotlib with matplotlib with matplotlib

fig_1 = plt.figure(figsize=(10, 10))
ax_1 = fig_1.add_subplot(111, projection='3d')
ax_1.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=20, alpha=0.5)

# 2. Random Sample from Triangle
# TODO: sample points randomly from mesh triangle
random_points = trimesh.sample.sample_surface(sphere, surface_sample_num)[0]
fig_2 = plt.figure(figsize=(10,10))
ax_2 = fig_2.add_subplot(111, projection='3d')
ax_2.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2])

# 3. Sample evenly.
# TODO: sample points uniformly from mesh triangle
uniform_points = trimesh.sample.sample_surface_even(sphere, surface_sample_num)[0]
fig_3 = plt.figure(figsize=(10,10))
ax_3 = fig_3.add_subplot(111, projection='3d')
ax_3.scatter(uniform_points[:, 0], uniform_points[:, 1], uniform_points[:, 2])
plt.show()