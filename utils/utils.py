import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
  
def get_graph_mat(n,size=1):
#    """ Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
#        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
#    """
    coords = size * np.random.uniform(size=(n,2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat
 
def update_q(q, dist_mat, state, action, alpha=0.012, gamma=0.9):
   immed_reward =dist_mat[state,action]  
   delayed_reward = q[action,:].max()
   q[state,action] += alpha * (immed_reward + gamma * delayed_reward - q[state,action]) # Q(s,a) = Q(s,a)+ alpha *(R(s,a) + gamma* max Q(s',a') - Q(s,a))
   return q

def find_the_best(dist_mat,n):
  q = np.zeros([n,n]) 
  epsilon = 1. 
  n_train = 8000 
  for i in range(n_train): 
    traj = [0] 
    state = 0
    possible_actions = [ dest for dest in range(n) if dest not in traj] 
    
    while possible_actions: 
      if np.random.random() < epsilon: 
        action = np.random.choice(possible_actions) 
      else:  
        best_action_index = q[state, possible_actions].argmax() 
        action = possible_actions[best_action_index] 
      q = update_q(q, dist_mat, state, action)
      traj.append(action) 
      state = traj[-1] 
      possible_actions = [ dest for dest in range(n) if dest not in traj] 
    
    action = 0
    q = update_q(q, dist_mat, state, action) 
    traj.append(0) 
    epsilon = 1. - i * 1/n_train
  traj = [0] 
  state = 0
  distance_travel = 0.
  possible_actions = [ dest for dest in range(n) if dest not in traj ] 
  while possible_actions: # until all destinations are visited 
     best_action_index = q[state, possible_actions].argmax() 
     action = possible_actions[best_action_index] 
     distance_travel += dist_mat[state, action] 
     traj.append(action) 
     state = traj[-1] 
     possible_actions = [ dest for dest in range(n) if dest not in traj ] 
  # Back to the first node
  action = 0
  distance_travel += dist_mat[state, action] 
  traj.append(action) 
  print('Best trajectory found:') 
  print(' -> '.join([str(b) for b in traj])) 
  print(f'Distance Travelled: {-1*distance_travel}')
  return(traj)

def plot_the_best(ans,coords):
  plt.scatter(coords[:,0],coords[:,1])
  for i in ans:
    plt.plot([coords[ans[i],0],coords[ans[i+1],0]],[coords[ans[i],1],coords[ans[i+1],1]])