import open3d as o3d
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from matplotlib import gridspec

if __name__ == "__main__":
    model = 'luigi'

    # load shape information
    vertices = np.loadtxt('../data/%s/%s_vertices.csv' % (model, model), delimiter=',', dtype=np.float32)
    vertex_normals = np.loadtxt('../data/%s/%s_vertex_normals.csv'% (model, model), delimiter=',',
                                dtype=np.float32)

    # first, let's create a grid
    bmax = vertices.max(axis=0)  # (3, )
    bmin = vertices.min(axis=0)  # (3, )
    bmax += 5  # add margin
    bmin -= 5  # add margin
    step = 3
    ###############################################################################################################
    ## TODO - (1) : create array named grids # (Gx, Gy, Gz, 3)

    ###############################################################################################################

    grids = grids.reshape(-1, 3)  # grids are now (Gx * Gy * Gz, 3)

    # visualize (1) - matplotlib version
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    bmin_all = bmin.min()
    bmax_all = bmax.max()
    ax.set_xlim(bmin_all, bmax_all)
    ax.set_ylim(bmin_all, bmax_all)
    ax.set_zlim(bmin_all, bmax_all)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')
    ax.scatter3D(vertices[..., 0], vertices[..., 1], vertices[..., 2], color='green', s=10, alpha=1.0)
    ax.scatter3D(grids[..., 0], grids[..., 1], grids[..., 2], color='blue', s=4, alpha=0.05)
    plt.show()

    ###############################################################################################################
    ## TODO - (2) : create array named sdf that corresponds grids # (Gx * Gy * Gz, 3)

    ###############################################################################################################

    vertices_norm = (vertices - bmin[None].astype(np.int32)) / step
    # visualize (2) - matplotlib version
    sdf_min = sdf.min()
    sdf_max = sdf.max()
    sdf = sdf.reshape(x_grid.shape)  # (G, G, G)
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('SDF sliced along Z-axis', fontweight='bold')
    gs = gridspec.GridSpec(nrows=2,
                           ncols=4,
                           height_ratios=[1, 1],
                           width_ratios=[1, 1, 1, 1]
                           )
    xx, yy = np.meshgrid(x_linspace, y_linspace)
    for i in range(1, 5):
        slice_depth = int(x_grid.shape[-1] * (i / 5))
        # visualize indicators
        ax = fig.add_subplot(gs[(i-1)], projection='3d')
        ax.view_init(elev=10)
        ax.scatter3D(vertices_norm[..., 0], vertices_norm[..., 1], vertices_norm[..., 2], color='green', s=2, alpha=0.6)
        zz = np.ones_like(xx) * slice_depth
        ax.plot_surface(xx, yy, zz, alpha=0.5)
        ax.set_title('Z=%d' %(slice_depth))
        # visualize sliced sdf
        ax = fig.add_subplot(gs[4 + (i-1)])
        ax.imshow(sdf[:, :, slice_depth], vmin=sdf_min, vmax=sdf_max)
        ax.scatter(vertices_norm[:, 0], vertices_norm[:, 1], color='grey', s=2, alpha=0.2)
    plt.show()

    ## TODO - (3) : now, verify calculated sdf
    # sdf to mesh
    sdf = sdf.reshape(*x_grid.shape)
    verts, faces, normals, _ = measure.marching_cubes(sdf, 0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh])
