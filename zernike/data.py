import numpy as np

from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh
import open3d as o3d

from sklearn.neighbors import BallTree

from collections import defaultdict

from moments import *

class PointCloud():
    def __init__(self, points, values=None, points_scaled=None):
        
        self.points = points
        
        if points_scaled is None:
            points_scaled = self.center_and_scale(points)
            
        self.points_scaled = points_scaled
        
        if values is None:
            values = np.array([1. for i in range(points.shape[0])])
            
        self.values = values
        
    def center_and_scale(self, points):
        center = points.mean(0)
        points = points - center
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points /= max_dist
        
        return points
    
    def plot(self, **fig_kwargs):
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points_scaled[:,0], 
                   self.points_scaled[:,1], 
                   self.points_scaled[:,2], 
                   c=self.values, s=1.)
        return ax

    def moments(self, order, agg='sum'):
        return moments_from_point_cloud(self.points_scaled, self.values, 
                                        order, agg=agg)

class PointGrid(PointCloud):
    def __init__(self, points, scale, sparse_grid=None, values=None, points_scaled=None):
        super().__init__(points, values, points_scaled)
        
        self.scale = scale
        
        if sparse_grid is None:
            occupied = (points/scale).astype(int)
            origin_index = occupied.min(0)
            sparse_grid = occupied - origin_index
            
        self.sparse_grid = sparse_grid
        
    @classmethod
    def point_grid_from_points(cls, points, values, scale, points_scaled=None):
        
        if points_scaled is None:
            center = points.mean(0)
            points_scaled = points - center
            max_dist = np.max(np.sqrt(np.sum(points_scaled**2, axis=1)))
            points_scaled /= max_dist
        
        hit = (points_scaled / scale).astype(int)
        _, u, inv = np.unique(hit, axis=0, return_index=True, return_inverse=True)
        
        occupied = hit[u]
        occ_vals = values[u]
        
        origin_index = occupied.min(axis=0)
        sparse_grid = occupied - origin_index
        
        return cls(points[u], scale, sparse_grid, occ_vals, points_scaled[u])
        
      
    @classmethod
    def point_grid_from_point_cloud(cls, point_cloud, scale):
        if type(scale) != np.ndarray:
            scale = np.array([scale]*3)
        return cls.point_grid_from_points(point_cloud.points, point_cloud.values, 
                                          scale, point_cloud.points_scaled)
        
    @classmethod
    def point_grid_from_mesh(cls, mesh, scale):
        if type(scale) != np.ndarray:
            scale = np.array([scale]*3)
        return cls.point_grid_from_points(mesh.points, mesh.values, scale, mesh.points_scaled)
    
    @classmethod
    def point_grid_from_voxel(cls, voxel):
        points = voxel.voxel.points
        values = voxel.values
        scale = voxel.voxel.scale
        sparse_indices = voxel.voxel.sparse_indices
        
        return cls(points, scale, sparse_indices, values, points)
    
    def moments(self, order, agg='sum'):
        return moments_from_point_grid(self.points_scaled, self.scale, self.values, 
                                        order, agg=agg)
    
class Mesh():
    def __init__(self, points, faces, normals, values=None, points_scaled=None):
    
        self.points = points
        
        if points_scaled is None:
            points_scaled = self.center_and_scale(points)
            
        self.points_scaled = points_scaled 
        
        if values is None:
            values = np.array([1. for i in range(points.shape[0])])
            
        self.values = values
        
        self.mesh = trimesh.Trimesh(self.points_scaled, faces, vector_normals=normals)
        
    def center_and_scale(self, points):
        center = points.mean(0)
        points = points - center
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points /= max_dist
        
        return points
    
    @classmethod
    def mesh_from_grid(cls, grid, isovalue=None, decimation=None):

        if isovalue is None: 
            isovalue = grid.mean()
        
        verts, faces, normals, values = measure.marching_cubes(grid, isovalue)
        
        mesh = cls(verts, faces, normals, values)

        if decimation is not None:
            mesh_o3d = mesh.mesh.as_open3d
            num_tri = np.array(mesh_o3d.triangles).shape[0]
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(int(num_tri*decimation))

            mesh = cls(verts, mesh_o3d.triangles, mesh_o3d.vertex_normals,
                        values, mesh_o3d.vertices)

        return mesh

    @classmethod 
    def mesh_from_point_grid(cls, point_grid, decimation=None):
        indices = point_grid.sparse_indices
        grid = np.zeros(tuple(indices.max(0)+1))

        for i in range(indices.shape[0]):
            inds = indices[i]
            x,y,z = inds 
            val = point_grid.values[i]
            grid[x,y,z] = val

        return cls.mesh_from_grid(grid, None, decimation)

    @classmethod
    def mesh_from_voxel(cls, voxel, decimation=None):

        indices = np.array(voxel.voxel.sparse_indices)
        grid = np.zeros(tuple(indices.max(0)+1))

        for i in range(indices.shape[0]):
            inds = indices[i]
            x,y,z = inds 
            val = voxel.values[i]
            grid[x,y,z] = val

        return cls.mesh_from_grid(grid, None, decimation)

    @classmethod
    def mesh_from_point_cloud(cls, point_cloud):
        
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud.points_scaled))
        pc.estimate_normals()

        distances = pc.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = avg_dist*2

        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                   pc,
                   o3d.utility.DoubleVector([radius, radius * 2]))
                
        mesh = cls(point_cloud.points, 
             np.array(mesh_o3d.triangles), 
             np.array(mesh_o3d.vertex_normals),
             point_cloud.values,
             point_cloud.points_scaled)
        
        trimesh.repair.fix_winding(mesh.mesh)
        trimesh.repair.fill_holes(mesh.mesh)
        
        return mesh    
    
    def plot(self, **fig_kwargs):
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111, projection='3d')
        
        m = Poly3DCollection(self.mesh.triangles)
        m.set_edgecolor('k')
        ax.add_collection3d(m)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1) 
        ax.set_zlim(-1, 1)
        return ax

    def moments(self, order, moment_type='volume', agg='sum'):
        return mesh_moment(self.mesh.vertices, self.mesh.faces,
                                self.values, order,
                                moment_type, agg)
        
    
class Voxel():
    def __init__(self, voxel, values=None):
        
        self.voxel = voxel
        
        if values is None:
            values = np.array([1. for i in range(voxel.points.shape[0])])
            
        self.values = values
        
        self.points = self.voxel.points
    
    @classmethod
    def voxel_from_mesh(cls, mesh, scale):
        
        voxel = mesh.mesh.voxelized(scale)
    
        tree = BallTree(mesh.mesh.vertices, leaf_size=15)
        
        distances, indices = tree.query(voxel.points, k=5)
    
        mask = (distances<scale)
        
        values = (mesh.values[indices]*mask).sum(-1) / np.clip(mask.sum(-1), 1, float('inf'))
        
        return cls(voxel, values)

    def plot(self, **fig_kwargs):
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[:,0], 
                   self.points[:,1], 
                   self.points[:,2], 
                   c=self.values, s=1.)
        return ax

    def moments(self, order, agg='sum', remove_empty=True):
        return moments_from_voxel(self, order, 
                                agg=agg, remove_empty=remove_empty)