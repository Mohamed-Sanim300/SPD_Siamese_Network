class LogEig_ts_function(Function):
  @staticmethod
  def forward(ctx,input_ts,u,S):
    s=S[:,:,:,:,:,:,0].log().diag_embed()
    ctx.save_for_backward(u,S,s)
    return u @ s @ u.transpose(-1,-2) 
    
  @staticmethod
  def backward(ctx,grad_output_ts):
    u,S,s = ctx.saved_tensors
    g=sym(grad_output_ts)
    S[S<0.0001]=0.0001
    P=S.clone()
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    dLdu= 2* g @ u @ s
    dLdS= (1/S[:,:,:,:,:,:,0]).diag_embed() @ u.transpose(-1,-2) @ g @ u
    idx=torch.arange(0,dLdS.size(-1), out=torch.LongTensor())
    k=dLdS[:,:,:,:,:,idx,idx].diag_embed()
    grad_input_ts=u @(( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    return grad_input_ts,dLdu,dLdS
class LogEig_ts(nn.Module):
  def __init__(self):
    super(LogEig_ts,self).__init__()

  def forward(self,input_ts,u,S):
    return LogEig_ts_function.apply(input_ts,u,S.unsqueeze(-1).expand(u.size()))
