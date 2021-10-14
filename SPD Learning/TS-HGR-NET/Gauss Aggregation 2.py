class Gauss_agg2_ts_function(Function):
  @staticmethod
  def forward(ctx,input_ts):
    ctx.save_for_backward(input_ts)
    #TS
    batch,seq,fingers,NS,joints,row,col=input_ts.size()
    input_ts=input_ts.reshape(batch,seq,fingers,NS*joints,row,col)
    mu=input_ts.mean(3)
    x=input_ts-mu.unsqueeze(3).expand(input_ts.size())
    cov=(x @ x.transpose(-1,-2)).mean(3)
    elt00 = cov + mu @ mu.transpose(-1,-2)
    elt01 = mu
    elt10 = mu.transpose(-1,-2)
    elt11 = torch.ones(batch,6,5,1,1)
    return torch.cat((torch.cat((elt00,elt01),-1),torch.cat((elt10,elt11),-1)),-2)
  @staticmethod
  def backward(ctx,grad_output_ts):
    input_ts=ctx.saved_tensors
    input_ts=input_ts[0]
    #TS
    batch,seq,fingers,NS,joints,row,col=input_ts.size()
    input_ts=input_ts.reshape(batch,seq,fingers,NS*joints,row).type(torch.DoubleTensor)
    B=torch.eye(row+1,row).reshape(1,1,row+1,row).expand(batch,seq,fingers,row+1,row).type(torch.DoubleTensor)
    b=torch.cat((torch.zeros(row),torch.ones(1))).reshape(1,1,1,row+1).expand(batch,seq,fingers,1,row+1)
    vect_one=torch.ones(batch,seq,fingers,NS*joints,1)
    g=sym(grad_output_ts).type(torch.DoubleTensor)
    gr=(2/(NS*4))* (input_ts @ B.transpose(-1,-2) + vect_one @ b) @ g @ B
    grad_input_ts=gr.reshape(batch,seq,fingers,NS,joints,row,col)    
    return grad_input_ts
class Gauss_agg2_ts(nn.Module):
  def __init__(self):
    super(Gauss_agg2_ts,self).__init__()

  def forward(self,input_ts):
    return Gauss_agg2_ts_function.apply(input_ts)
