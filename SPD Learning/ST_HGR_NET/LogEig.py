class LogEig_st_function(Function):
  @staticmethod
  def forward(ctx,input_st,u,S):
    #ST
    s=S[:,:,:,:,0].log().diag_embed()
    ctx.save_for_backward(u,S,s)
    return u @ s @ u.transpose(-1,-2)
    
  @staticmethod
  def backward(ctx,grad_output_st):
    u,S,s = ctx.saved_tensors
    g=sym(grad_output_st)
    P=S.clone()
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    dLdu= 2* g @ u @ s
    dLdS= (1/S[:,:,:,:,0]).diag_embed() @ u.transpose(-1,-2) @ g @ u
    idx=torch.arange(0,dLdS.size(3), out=torch.LongTensor())
    k=dLdS[:,:,:,idx,idx].diag_embed()
    grad_input_st=u @(( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    
    return grad_input_st,dLdu,dLdS
class LogEig_st(nn.Module):
  def __init__(self):
    super(LogEig_st,self).__init__()

  def forward(self,input_st,u,S):
    return LogEig_st_function.apply(input_st,u,S.unsqueeze(-1).expand(u.size()))
