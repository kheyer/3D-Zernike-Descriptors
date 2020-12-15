import numpy as np 

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