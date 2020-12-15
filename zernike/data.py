import numpy as np

from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh
import open3d as o3d

from sklearn.neighbors import BallTree

from collections import defaultdict


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
    def mesh_from_grid(cls, grid, isovalue):
        
        verts, faces, normals, values = measure.marching_cubes(grid, isovalue)
        
        return cls(verts, faces, normals, values)
    
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
             np.array(mesh.triangles), 
             np.array(mesh.vertex_normals),
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

    @classmethod
    def voxel_from_point_cloud(cls, point_cloud):
        # TODO
        raise NotImplementedError

    
    def plot(self, **fig_kwargs):
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[:,0], 
                   self.points[:,1], 
                   self.points[:,2], 
                   c=self.values, s=1.)
        return ax