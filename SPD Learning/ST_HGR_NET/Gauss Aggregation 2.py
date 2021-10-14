import torch
from torch import nn
from torch.autograd import Function
from util import

class Gauss_agg2_st_function(Function):
  
  @staticmethod
  def forward(ctx,x0,x1,x2,x3,x4,x5):
    ctx.save_for_backward(x0,x1,x2,x3,x4,x5)
    input_st = [x0,x1,x2,x3,x4,x5]

    mu = torch.zeros(x0.size(0),6,x0.size(1),x0.size(3),1)
    cov = torch.zeros(x0.size(0),6,x0.size(1),x0.size(3),x0.size(3))
    for s in range(6):
      batch,fingers,nb_frames,row,col = input_st[s].size()
      mu[:,s] = input_st[s].mean(2)
      x = input_st[s]-mu[:,s].unsqueeze(2).expand(batch,fingers,nb_frames,row,col)
      cov[:,s] = (x @ x.transpose(-1,-2)).mean(2)
    elt00 = cov + mu @ mu.transpose(-1,-2)
    elt01 = mu
    elt10 = mu.transpose(-1,-2)
    elt11 = torch.ones(batch,6,5,1,1)
    return torch.cat((torch.cat((elt00,elt01),-1),torch.cat((elt10,elt11),-1)),-2)
  
  @staticmethod
  def backward(ctx,grad_output_st):
    x0,x1,x2,x3,x4,x5 = ctx.saved_tensors
    input_st = [x0,x1,x2,x3,x4,x5]
    grad_input_st = []
    batch,fingers,nb_frames,row,col = x0.size()
    B = torch.eye(row+1,row).reshape(1,1,row+1,row).expand(batch,fingers,row+1,row)
    b = torch.cat((torch.zeros(row),torch.ones(1))).reshape(1,1,1,row+1).expand(batch,fingers,1,row+1)
    g = sym(grad_output_st)
    #ST
    for s in range(6):
      nb_frames = input_st[s].size(2)
      x = input_st[s].squeeze(-1)
      vect_one = torch.ones(batch,fingers,nb_frames,1)
      gr = (2/(nb_frames))* (x @ B.transpose(-1,-2) + vect_one @ b) @ g[:,s] @ B
      grad_input_st.append(gr.unsqueeze(-1))
    return grad_input_st[0],grad_input_st[1],grad_input_st[2],grad_input_st[3],grad_input_st[4],grad_input_st[5]
  
  
class Gauss_agg2_st(nn.Module):
  def __init__(self):
    super(Gauss_agg2_st,self).__init__()

  def forward(self,input_st):
    nb_frames = int(input_st.size(2)/3)
    l_sp = [len(sequence(nb_frames)[s]) for s in range(6)]
    x0,x1,x2,x3,x4,x5 = input_st.split(l_sp,2)
    return Gauss_agg2_st_function.apply(x0,x1,x2,x3,x4,x5)
