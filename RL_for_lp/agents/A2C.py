import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
  def __init__(self,T,e):
    super().__init__()
    self.T=T
    self.e=e
    self.theta1=nn.Linear(self.e,self.e)
    self.theta2=nn.Linear(self.e,self.e)
    self.theta3=nn.Linear(self.e,self.e)
    self.theta4=nn.Linear(1,self.e)
    self.theta5=nn.Linear(2*self.e,1)
    self.theta6=nn.Linear(self.e,self.e)
    self.theta7=nn.Linear(self.e,self.e)

  def forward(self,xv,W):
    n=xv.shape[1]
    e=xv.shape[2]
    s1=self.theta1(xv)
    mu=torch.zeros(1,n,e)
    adj=torch.where(W>0,torch.ones_like(W) ,torch.zeros_like(W))
    s2=self.theta2(adj.matmul(mu))
    s3=self.theta3( torch.sum( F.relu(self.theta4(W.unsqueeze(3))) , dim=1) )
    for i in range(self.T):
      mu=F.relu(s1+s2+s3)
      s2=self.theta2(adj.matmul(mu))    
    soft=F.softmax(mu,dim=1)
    p=torch.sum(soft,dim=2)
    #max_p=torch.max(p)
    #action=torch.argmax(p)
    return (p)
  

class Critic(nn.Module):

  def __init__(self,T,e):
    super().__init__()
    self.T=T
    self.e=e
    self.theta1=nn.Linear(self.e,self.e)
    self.theta2=nn.Linear(self.e,self.e)
    self.theta3=nn.Linear(self.e,self.e)
    self.theta4=nn.Linear(1,self.e)
    self.theta5=nn.Linear(2*self.e,1)
    self.theta6=nn.Linear(self.e,self.e)
    self.theta7=nn.Linear(self.e,self.e)
    
  def forward(self,xv,W,action):
    n=xv.shape[1]
    e=xv.shape[2]
    s1=self.theta1(xv)
    mu=torch.zeros(1,n,e)
    adj=torch.where(W>0,torch.ones_like(W) ,torch.zeros_like(W))
    s2=self.theta2(adj.matmul(mu))
    s3=self.theta3( torch.sum( F.relu(self.theta4(W.unsqueeze(3))) , dim=1) )
    for i in range(self.T):
      mu=F.relu(s1+s2+s3)
      s2=self.theta2(adj.matmul(mu))
#q-values
    global_action=self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1,n,1))
    local=self.theta7(mu)
    out=F.relu(torch.cat([global_action,local],dim=2))  
    est_rew=self.theta5(out).squeeze(dim=2)
    q=est_rew[0][action]
    return (q)
