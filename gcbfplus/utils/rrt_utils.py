

import numpy as np
from rrt_algorithms.rrt.rrt import RRT 
from rrt_algorithms.search_space.search_space import SearchSpace


def gen_obstacles(obs, obs_w, obs_h):

    obs_len = np.hstack((obs_w[:, None], obs_h[:, None])) 
    Obstacles = np.zeros((obs.shape[0], 4))
    for i in range(obs.shape[0]):
        Obstacles[i, :] = [obs[i, 0] - obs_len[i, 0], obs[i, 1] - obs_len[i, 1], obs[i, 0] + obs_len[i, 0], obs[i, 1] + obs_len[i, 1]]
        
    X_dimensions = np.array([(0, 5), (0, 5)])  
    return X_dimensions, Obstacles

def find_path(obs_c, obs_w, obs_l, x_init, x_goal, q=0.01, r=0.01, max_samples=100000, prc=0.1):
    
    
    x_init = (x_init[0].item(), x_init[1].item())
    x_goal = (x_goal[0].item(), x_goal[1].item())
    X_dimensions, Obstacles = gen_obstacles(obs_c, obs_w.squeeze(), obs_l.squeeze())
    
    X = SearchSpace(X_dimensions, Obstacles)

    
    rrt = RRT(X, q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()
    return path
