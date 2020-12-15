import numpy as np 
from sklearn.neighbors import BallTree

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
    y2 = p2[:,0]
    
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
                    else:
                        moments[:,r,s,t] = moment
                    
    return moments

