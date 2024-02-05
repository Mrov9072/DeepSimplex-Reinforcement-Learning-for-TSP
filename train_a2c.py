import torch
import numpy as np
import random
from torch import optim
from Models.A2C import Actor,Critic
import matplotlib.pyplot as plt

SEED=1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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
   
MIN_EPSILON = 0.1
EPSILON_DECAY_RATE = 6e-4

def find_the_best(e):
  c=1
  for episode in range(e):
    print('Episode number :{}  '.format(c),end="")
    sum_actor_loss=sum_critic_loss=0
    current_state=[0]                                                          
    next_state=[0]                                                             
    while len(current_state)<n:
      actor_prob=actor(tensor(current_state),dist_mat)
      listed_prob=actor_prob.flatten().tolist()
      epsilon = max(MIN_EPSILON, (1-EPSILON_DECAY_RATE)**e)
      if epsilon>=random.random():
        p_a=random.choice(listed_prob)
        action=listed_prob.index(p_a)
        while action in current_state:
          p_a=random.choice(listed_prob)
          action=listed_prob.index(p_a)       
      else:
        o=actor_prob.argsort(descending=True)
        for action in o[0]:
          if action not in current_state:
            p_a=listed_prob[action]
            break
      current_action=torch.tensor(action)                                      
      next_state.append(int(action))                                            
      p=actor(tensor(next_state),dist_mat)
      listed_prob=p.flatten().tolist()
      o=p.argsort(descending=True)
      for action in o[0]:
        if action!=current_action:                                            
          next_action=action                                                    
          break
      r=-dist_mat[0,next_state[-2],next_state[-1]]                              
      Q_current= critic(tensor(current_state),dist_mat,current_action)          
      Q_next = critic(tensor(next_state),dist_mat,next_action)                  
      adv=r+gamma*Q_next - Q_current                                            
      log_prob=torch.log(torch.tensor(p_a))                                                                              
      actor_loss=-log_prob*Q_current                                           
      actor_loss.backward(retain_graph=True)
      optimizer_actor.step()
      lr_scheduler_actor.step()
      optimizer_actor.zero_grad()
      sum_actor_loss+=actor_loss
      critic_loss=-adv*Q_current                                               
      critic_loss.backward()
      optimizer_critic.step()
      lr_scheduler_critic.step()
      optimizer_critic.zero_grad()
      sum_critic_loss+=critic_loss
      current_state.append(next_state[-1])
    current_state.append(current_state[0])
    total_dist=0 
    for i in range(len(current_state)-1):
      total_dist+=dist_mat[0,current_state[i],current_state[i+1]]
    print('total_dist-',total_dist,end="")
    c+=1
    x[total_dist]=current_state
    list_actor_loss.append(sum_actor_loss)
    list_critic_loss.append(sum_critic_loss)
    print("  Actor loss {:.2f} | | critic loss {:.2f} ".format(sum_actor_loss,sum_critic_loss))
    
def plot_the_best(ans):
  plt.scatter(coords[:,0],coords[:,1])
  for i in ans:
    plt.plot([coords[ans[i],0],coords[ans[i+1],0]],[coords[ans[i],1],coords[ans[i+1],1]])
  plt.show()
    
gamma=0.99
actor=Actor(1,3)
critic=Critic(1,3)
optimizer_actor = optim.Adam(actor.parameters(), lr=5e-3)
lr_scheduler_actor = optim.lr_scheduler.ExponentialLR(optimizer_actor, gamma=1. - 2e-5)
optimizer_critic = optim.Adam(critic.parameters(), lr=5e-3)
lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(optimizer_critic, gamma=1. - 2e-5)
n=10
coords=np.load('data//coords_a2c.npy')
dist_mat = np.load('data//dist_matrix_a2c.npy')
dist_mat= torch.FloatTensor(dist_mat)
dist_mat=-dist_mat.unsqueeze(0)
global x,list_actor_loss, list_critic_loss
list_actor_loss=[]
list_critic_loss=[]
e=8000
x=dict()
find_the_best(e)
distances=list(x.keys())
opt=distances[0]
for i in range(len(distances)):
  if distances[i]<opt:
    opt=distances[i]
traj=x[opt]
plot_the_best(traj)
print('shortest distance',opt)