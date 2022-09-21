import numpy as np
from matplotlib import pyplot as plt
from utils import find_correspondence, find_registration, random_sampling

try:
    import open3d as o3d
    model_pc = np.asarray(o3d.io.read_point_cloud('../data/bun000.ply').points).transpose()
    scene_pc = np.asarray(o3d.io.read_point_cloud('../data/bun045.ply').points).transpose()
except:
    model_pc = np.loadtxt('../data/model_pc.txt')
    scene_pc = np.loadtxt('../data/scene_pc.txt')

###########################################################
# You can modify these values
max_iter = 20
dist_thres = 0.1
###########################################################

mse = np.zeros(max_iter)
R = np.eye(3)
t = np.zeros((3, 1))

# visualize initial points
vis_indices_m = np.random.permutation(np.arange(0, model_pc.shape[1]))[:5000]
vis_indices_s = np.random.permutation(np.arange(0, scene_pc.shape[1]))[:5000]
fig_1 = plt.figure(figsize=(10, 10))
fig_1.suptitle('Before ICP', fontweight='bold')
ax_1 = fig_1.add_subplot(projection='3d')
ax_1.scatter3D(model_pc[0, vis_indices_m], model_pc[1, vis_indices_m], model_pc[2, vis_indices_m])
ax_1.scatter3D(scene_pc[0, vis_indices_s], scene_pc[1, vis_indices_s], scene_pc[2, vis_indices_s], color='red')

# ICP
for it in range(max_iter):
    sampled_model_pc = random_sampling(model_pc, 5000)
    sampled_scene_pc = random_sampling(scene_pc, 5000)
    corr = find_correspondence(sampled_model_pc, sampled_scene_pc, R, t, dist_thres)
    R, t, mse[it] = find_registration(sampled_model_pc, sampled_scene_pc, corr)

# visualize results of ICP
fig_2 = plt.figure(figsize=(10, 10))
fig_2.suptitle('After ICP', fontweight='bold')
ax_2 = fig_2.add_subplot(projection='3d')
ax_2.scatter3D(model_pc[0, vis_indices_m], model_pc[1, vis_indices_m], model_pc[2, vis_indices_m])
rotated_scene = np.matmul(R, scene_pc) + t
ax_2.scatter3D(rotated_scene[0, vis_indices_s], rotated_scene[1, vis_indices_s], rotated_scene[2, vis_indices_s], color='red')


fig_3 = plt.figure()
fig_3.suptitle('MSE for each iter.', fontweight='bold')
ax_3 = fig_3.add_subplot()
ax_3.scatter(np.arange(max_iter), mse)
ax_3.set_xlabel('Iteration', fontweight='bold')
ax_3.set_ylabel('MSE', fontweight='bold')

plt.show()
