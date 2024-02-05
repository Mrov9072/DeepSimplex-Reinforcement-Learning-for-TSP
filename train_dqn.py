import numpy as np
import random
import torch
from Models.DQN import RLnetwork 
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

SEED=1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def plot_the_best(ans):
  plt.scatter(coords[:,0],coords[:,1])
  for i in ans:
    plt.plot([coords[ans[i],0],coords[ans[i+1],0]],[coords[ans[i],1],coords[ans[i+1],1]])
  plt.show()

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
def find_the_best(e):
  loss=0
  epsilon=1.
  for episode in range(e):
    state=0
    path=[state]
    for i in range(n-1):
      if epsilon > random.random():
        action_prob=model(tensor(path))
        q_s_a=random.choice(action_prob)
        action=action_prob.tolist().index(q_s_a)
        while action in path:
          q_s_a=random.choice(action_prob)
          action=action_prob.tolist().index(q_s_a)
      else:
        action_prob=model(tensor(path))
        sorted_prob=action_prob.argsort(descending=True)
        for i in sorted_prob:
          if i not in path:
            action=i
            q_s_a=action_prob[i]
            break
      next_action_prob=model(tensor([action]))
      sorted_prob=next_action_prob.argsort(descending=True)
      target=dist_mat[state,action]+0.9*next_action_prob[sorted_prob[0]]
      loss=F.mse_loss(q_s_a,target)
      loss.backward(retain_graph=True)
      torch.autograd.set_detect_anomaly(True)
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      path.append(int(action))
      state=int(action)
    epsilon=1-episode/e
    path.append(path[0])
    total_dist=0
    for i in range(len(path)-1):
      total_dist+=dist_mat[path[i],path[i+1]]
    print('episode number {} total distance {}'.format(episode,-total_dist))
    if episode==0:
      x[-total_dist]=path
      m.append(-total_dist)
    if episode>0:
      if -total_dist<m[-1]:
        x[-total_dist]=path
        m.append(-total_dist)
    
coords = np.load('data//coords_dqn.npy')
dist_mat=np.load('data//dist_matrix_dqn.npy')
n=5
model=RLnetwork(64,32,32,32,n)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1. - 2e-5)
global x,m
x=dict()
m=[]
dist_mat*=-1
find_the_best(8000)
ans=list(x.values())[-1]
plot_the_best(ans)