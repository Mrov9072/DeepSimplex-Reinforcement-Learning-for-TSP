import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torch import optim
from Models.Graph_Embeded_DQN import RLnetwork
import torch.nn.functional as F

def tensor(ans):
  xv=[]
  for i in range(n):
    if i not in ans:
      xv.append([0,coords[i,0],coords[i,1]])
    else:
      xv.append([1,coords[i,0],coords[i,1]])
  xv=torch.FloatTensor(xv)
  xv=xv.unsqueeze(0)  
  return (xv)
   
def find_the_best(episodes):
  loss=0
  epsilon=1
  for episode in range(episodes):
    state=0
    path=[state]
    
    while len(path)<n:
      if epsilon > random.random():
        current_est_rew=model(tensor(path),dist_mat)
        q_value=random.choice(current_est_rew)
        action=current_est_rew.tolist().index(q_value)
        while action in path:
          q_value=random.choice(current_est_rew)
          action=current_est_rew.tolist().index(q_value)
      else:
        current_est_rew=model(tensor(path),dist_mat)
        sorted_est_rew=current_est_rew.argsort(descending=True)
        for action in sorted_est_rew:
          if action not in path:
            q_value=current_est_rew[action]
            break
      
      next_est_rew=model(tensor(path),dist_mat)
      sorted_est_rew=next_est_rew.argsort(descending=True)
      target=dist_mat[0,state,action] + 0.9*next_est_rew[sorted_est_rew[0]]
      
      path.append(int(action))
      state=action

      loss=F.mse_loss(q_value,target)
      loss.backward(retain_graph=True)
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()

    epsilon=1-episode/episodes
    path.append(path[0])
    total_dist=0
    for i in range(len(path)-1):
      total_dist+=dist_mat[0,path[i],path[i+1]]
    print('episode number {} total distance {}'.format(episode,-total_dist))

    if episode==0:
      x[-total_dist]=path
      m.append(-total_dist)
    if episode>0:
      if -total_dist<m[-1]:
        x[-total_dist]=path
        m.append(-total_dist)
    
def plot_the_best(path):
  plt.scatter(coords[:,0],coords[:,1])
  for i in path:
    plt.plot([coords[path[i],0],coords[path[i+1],0]],[coords[path[i],1],coords[path[i+1],1]])
  plt.show()
  
global x,m
n=4
model=RLnetwork(1,3)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1. - 2e-5)
x=dict()
m=[]
coords=np.load('data//coords_graph_embeded_dqn.npy')
dist_mat=np.load('data//dist_matrix_graph_embeded_dqn.npy')
dist_mat*=-1
dist_mat= torch.FloatTensor(dist_mat)
dist_mat=dist_mat.unsqueeze(0)
find_the_best(10000)
path=list(x.values())[-1]
plot_the_best(path)
print('shortest distance',list(x)[-1])