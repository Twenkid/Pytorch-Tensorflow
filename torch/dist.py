import torch
import torch.nn.functional as F
#Compute distances: eucledian, cosine

def normalize(x, axis=-1):  
  """Performs L2-Norm."""
  num = x
  denom = torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12
  return num / denom
  
def euclidean_dist(x, y):
  """Computes Euclidean distance."""
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
  yy = torch.pow(x, 2).sum(1, keepdim=True).expand(m, m).t()
  dist = xx + yy - 2 * torch.matmul(x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()
  return dist
  
def cosine_dist(x, y):
  """Computes Cosine Distance."""
  x = F.normalize(x, dim=1)
  y = F.normalize(y, dim=1)
  dist = 2 - 2 * torch.mm(x, y.t())
  return dist

import numpy as np

def is_singular(A):
  det = np.linalg.det(A)
  if det == 0: return True
  else:
       return False
  A = np.array([[1, 2], [2, 4]])
  print(is_singular(A)) # True
