import numpy as np 
from sklearn.neighbors import BallTree
from scipy.special import factorial
import time

from numba import jit 

@jit(nopython=True, parallel=True)
def Dabc(CFs, order):
    n_facets = CFs.shape[0]
    D = np.zeros((n_facets, order+1, order+1, order+1))
    # C0 = CFs[:,0]
    C1 = CFs[:,1]
    C2 = CFs[:,2]

    for i in range(order+1):
        for j in range(order+1):
            for k in range(order+1):
                for ii in range(i+1):
                    for jj in range(j+1):
                        for kk in range(k+1):
                            D[:,i,j,k] += C1[:,ii,jj,kk] * C2[:,i-ii, j-jj, k-kk]

    return D

def trinomial(i, j, k):
    output = factorial(i+j+k)
    output /= factorial(i)
    output /= factorial(j)
    output /= factorial(k)
    return output

def moments_from_voxel(voxel, order, agg='sum', remove_empty=True):

    points = voxel.voxel.points 
    scale = voxel.voxel.scale 
    values = voxel.values

    if remove_empty:
        points = points[values != 0.]
        values = values[values != 0.]

    return moments_from_point_grid(voxel.voxel.points, 
                                   voxel.voxel.scale, 
                                   voxel.values, order, agg=agg)

def moments_from_point_grid(points, diffs, values, order, agg='sum'):
        
    x1 = points[:,0] - diffs[0]/2
    x2 = points[:,0] + diffs[0]/2
    
    y1 = points[:,1] - diffs[1]/2
    y2 = points[:,1] + diffs[1]/2
    
    z1 = points[:,2] - diffs[2]/2
    z2 = points[:,2] + diffs[2]/2
    
    if agg=='sum':
        moments = np.zeros((order+1, order+1, order+1))
    else:
        moments = np.zeros((x1.shape[0], order+1, order+1, order+1))
    
    for r in range(order+1):
        for s in range(order+1):
            for t in range(order+1):
                if r+s+t <= order:
                    x_term = (x2**(r+1) - x1**(r+1))
                    y_term = (y2**(s+1) - y1**(s+1))
                    z_term = (z2**(t+1) - z1**(t+1))

                    moment = values * x_term * y_term * z_term / ((r+1) * (s+1) * (t+1))
                    
                    if agg=='sum':
                        moment = moment.sum()
                        moments[r,s,t] = moment
                    elif agg=='mean':
                        moment = moment.mean()
                        moments[r,s,t] = moment

                    else:
                        moments[:,r,s,t] = moment
                    
    return moments

def moments_from_point_cloud(points, values, order, agg='sum'):
    
    tree = BallTree(points, leaf_size=15)
    distances, indices = tree.query(points, k=2)

    distances = distances[:,1].reshape(-1,1)
            
    diffs = distances.mean(0)
        
    p1 = points - diffs/2
    p2 = points + diffs/2
    
    x1 = p1[:,0]
    x2 = p2[:,0]
        
    y1 = p1[:,1]
    y2 = p2[:,1]
    
    z1 = p1[:,2]
    z2 = p2[:,2]
    
    
    if agg=='sum':
        moments = np.zeros((order+1, order+1, order+1))
    else:
        moments = np.zeros((x1.shape[0], order+1, order+1, order+1))
    
    for r in range(order+1):
        for s in range(order+1):
            for t in range(order+1):
                if r+s+t <= order:
                    x_term = (x2**(r+1) - x1**(r+1))
                    y_term = (y2**(s+1) - y1**(s+1))
                    z_term = (z2**(t+1) - z1**(t+1))

                    moment = values * x_term * y_term * z_term / ((r+1) * (s+1) * (t+1))
                    
                    if agg=='sum':
                        moment = moment.sum()
                        moments[r,s,t] = moment
                    elif agg=='mean':
                        moment = moment.mean()
                        moments[r,s,t] = moment
                    else:
                        moments[:,r,s,t] = moment
                    
    return moments

def mesh_moment(verts, faces, values, order, moment='volume', agg='sum'):

    start = time.time()

    n_points = verts.shape[0]
    n_facets = faces.shape[0]
        
    tri_array = np.zeros((order+1, order+1, order+1))

    for i in range(order+1):
        for j in range(order-i+1):
            for k in range(order-i-j+1):
                tri_array[i,j,k] = trinomial(i,j,k)

    tri_time = time.time() - start 
    print(f'tri time: {tri_time}')

    monomial_array = np.zeros([n_points, order + 1, order + 1, order + 1])

    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]

    for i in range(order+1):
        for j in range(order-i+1):
            for k in range(order-i-j+1):
                monomial_array[:,i,j,k] = tri_array[i,j,k] * (x**i) * (y**j) * (z**k) * values
                
    monomial_time = time.time() - tri_time - start
    print(f'monomial time: {monomial_time}')

    CFs = monomial_array[faces]
    C0 = CFs[:,0]

    D = Dabc(CFs, order)
    
    # D = np.zeros((n_facets, order+1, order+1, order+1))
    # C0 = CFs[:,0]
    # C1 = CFs[:,1]
    # C2 = CFs[:,2]

    # for i in range(order+1):
    #     for j in range(order+1):
    #         for k in range(order+1):
    #             for ii in range(i+1):
    #                 for jj in range(j+1):
    #                     for kk in range(k+1):
    #                         D[:,i,j,k] += C1[:,ii,jj,kk] * C2[:,i-ii, j-jj, k-kk]

    d_time = time.time() - monomial_time - start
    print(f'Dabc time: {d_time}')
                            
    S = np.zeros((n_facets, order+1, order+1, order+1))
    for i in range(order+1):
        for j in range(order-i+1):
            for k in range(order-i-j+1):
                for ii in range(i+1):
                    for jj in range(j+1):
                        for kk in range(k+1):
                            S[:,i,j,k] += C0[:,ii,jj,kk] * D[:,i-ii, j-jj, k-kk]

    s_time = time.time() - d_time - start
    print(f'Sijk time: {s_time}')
                            
    i, j, k = np.mgrid[0:order + 1, 0:order + 1, 0:order + 1]
    coef = factorial(i) * factorial(j) * factorial(k) / factorial(i + j + k + 2)
    S *= coef[None]
    
    if moment == 'volume':
                            
        volumes = np.linalg.det(verts[faces].transpose(0,2,1)) # actually 6x volume
        # volumes = np.sign(volumes)*volumes

        moments_array = (volumes[:,None,None,None]*S)
        
        moments_array /= (i+j+k+3)
    
    else:
        tri = verts[faces]
        areas = np.linalg.norm(np.cross(tri[:,0]-tri[:,2], tri[:,1]-tri[:,2]), 2, axis=-1) # actually 2x area
        
        moments_array = (areas[:,None,None,None]*S)

    if agg=='sum':
        moments_array = moments_array.sum(0)
    elif agg=='mean':
        moments_array = moments_array.mean(0)
    else:
        pass  

    f_time = time.time() - start
    print(f'final time: {f_time}')
    
    return moments_array
