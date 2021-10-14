class ReEig_ts_function(Function):
  @staticmethod
  def forward(ctx,input_ts,eps):
    eps=torch.tensor(eps)
    #TS
    u,S,v=input_ts.svd()
    ctx.save_for_backward(u,S.clone(),eps)
    S[S<eps]=eps
    return u @ S.diag_embed() @ u.transpose(-1,-2),u,S 
  
  @staticmethod
  def backward(ctx,grad_output_ts,grad_u,grad_S):
    u,S,eps= ctx.saved_tensors
    #TS
    P = S.unsqueeze(-1).expand(u.size())
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    Q=torch.ones(S.size())
    Q[S<eps]=0
    Q=Q.diag_embed()
    g=sym(grad_output_ts) 
    S[S<eps]=eps
    dLdu= 2* g @ u @ S.diag_embed()
    dLdS= Q @ u.transpose(-1,-2) @ g @ u
    idx=torch.arange(0,dLdS.size(-1), out=torch.LongTensor())
    k=dLdS[:,:,:,:,:,idx,idx].diag_embed()
    grad_input_ts=u @ (( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    return grad_input_ts,None
class ReEig_ts(nn.Module):
  def __init__(self,eps=10**(-4)):
    super(ReEig_ts,self).__init__()
    self.eps=eps

  def forward(self,input_ts):
    return ReEig_ts_function.apply(input_ts,self.eps)
