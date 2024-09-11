import torch
from torch import nn
from torch.func import functional_call, vmap, hessian
import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(404)
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.fc1=nn.Linear(1,1000)
    self.fc2=nn.Linear(1000,1)
    self.af=nn.Tanh()
  def forward(self, x):
    x=self.fc1(x)
    x=self.af(x)
    x=self.fc2(x)
    return x.squeeze(-1)

net = Model()

batch_size=100

targets = torch.randn(batch_size)
inputs = torch.randn(batch_size, 1)
params = dict(net.named_parameters())

def fcall(params, inputs):
  outputs = functional_call(net, params, inputs)
  return outputs

def loss_fn(outputs, targets):
  return torch.mean((outputs - targets)**2, dim=0)

def compute_loss(params, inputs, targets):
  outputs = vmap(fcall, in_dims=(None,0))(params, inputs) #vectorize over batch
  return loss_fn(outputs, targets)
  
def compute_hessian_loss(params, inputs, targets):
  return hessian(compute_loss, argnums=(0))(params, inputs, targets)

loss = compute_loss(params, inputs, targets)
print(loss)

hess = compute_hessian_loss(params, inputs, targets)
key=list(params.keys())[0] #take weight in first layer as example key
h = hess[key][key]
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print(params.keys())

for p in params.keys():
  m = hess[p][p].squeeze()
  if m.dim() > 0:
    diag = torch.diagonal(m)
    diag_norm = torch.norm(diag)
  else:
    diag = torch.tensor([m])
    diag_norm = torch.norm(diag)
  out = torch.tensor([])

  for p_s in params.keys():
    res = hess[p][p_s]
    if p_s == p:
      if m.dim() > 0:
        res = hess[p][p_s].squeeze() - torch.diag(diag)
      else:
        res = torch.tensor([hess[p][p_s].squeeze()]) - diag
    if out.numel() == 0:
      out = res.reshape(-1)
    else:
      out = torch.cat((out, res.reshape(-1)))   
  print(diag_norm/torch.norm(out))

#print(hess) #Hessian of loss w.r.t first weight (shape [16, 1, 16, 1]).