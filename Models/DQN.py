from torch import nn 
import torch.nn.functional as F 


class RLnetwork(nn.Module):
  def __init__(self,hidden1_size,hidden2_size,hidden3_size,hidden4_size,n):
    super().__init__()
    self.layer1=nn.Linear(3,hidden1_size)
    self.hidden1=nn.Linear(hidden1_size,hidden2_size)
    self.hidden2=nn.Linear(hidden2_size,hidden3_size)
    self.hidden3=nn.Linear(hidden3_size,hidden4_size)
    self.hidden4=nn.Linear(hidden4_size,hidden4_size)
    self.out=nn.Linear(hidden4_size,n)

  def forward(self,state):
    layer1=self.layer1(state)
    layer2=F.relu(self.hidden1(layer1))
    layer3=F.relu(self.hidden2(layer2))
    layer4=F.relu(self.hidden3(layer3))
    layer5=F.relu(self.hidden4(layer4))
    output=self.out(layer5)
    return output.sum(dim=1).squeeze(dim=0)
