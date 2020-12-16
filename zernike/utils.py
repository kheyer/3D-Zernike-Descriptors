import numpy as np 
from collections import defaultdict

def rotation_matrix(alpha, beta, gamma):
    yaw = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])
    
    pitch = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])
    
    roll = np.array([[1, 0, 0],
                     [0, np.cos(gamma), -np.sin(gamma)],
                     [0, np.sin(gamma), np.cos(gamma)]])
    
    rotation = yaw@pitch@roll
    
    return rotation

def omega_to_norm(omegas, order):
    packed = defaultdict(list)

    for k,v in omegas.items():
        n,l,m = k

        packed[(n,l)].append(v)

    F = np.zeros((order+1, order+1))

    for k,v in packed.items():
        n,l = k
        F[n,l] = np.linalg.norm(v, ord=2)
        
    return F
