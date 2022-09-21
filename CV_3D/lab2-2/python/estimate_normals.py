import open3d as o3d
import numpy as np


if __name__ == "__main__":
    model = 'luigi'

    # load shape information
    vertices = np.loadtxt('../data/%s/%s_vertices.csv' % (model, model), delimiter=',', dtype=np.float32)
    vertex_normals = np.loadtxt('../data/%s/%s_vertex_normals.csv'% (model, model), delimiter=',',
                                dtype=np.float32)

    # construct point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.normals = o3d.utility.Vector3dVector(vertex_normals)
    pcd.colors = o3d.utility.Vector3dVector(np.array([0, 1, 0])[None].repeat(vertices.shape[0], axis=0))

    # visualization
    # press n to visualize normal vectors
    # press + or - to increase or decrease point size
    o3d.visualization.draw_geometries([pcd])

    # calculate vertex_normals from vertices and compare to ground truth normals

    # K for K-nearest neighbor
    K = 10
    N, _ = vertices.shape

    ###############################################################################################################
    ## TODO : calculate vertex normals from vertices
    ## TODO : You need to create array named vertex_normals_hat  whose shape is (N, 3) (same as 'vertex_normals')

    ###############################################################################################################

    # draw new point cloud with computed vertex_normals
    pcd.normals = o3d.utility.Vector3dVector(vertex_normals_hat)
    o3d.visualization.draw_geometries([pcd])
