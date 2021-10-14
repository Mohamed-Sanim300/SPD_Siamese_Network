class ReEig_st_function(Function):
  @staticmethod
  def forward(ctx,input_st,eps):
    eps=torch.tensor(eps)
    #ST
    u,S,v=input_st.svd()
    ctx.save_for_backward(u,S.clone(),eps)
    S[S<eps]=eps
    return u @ S.diag_embed() @ u.transpose(-1,-2),u,S
  
  @staticmethod
  def backward(ctx,grad_output_st,grad_u,grad_S):
    u,S,eps = ctx.saved_tensors
    eps=float(eps)
    #ST
    P = S.unsqueeze(-1).expand(u.size())
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    Q=torch.ones(S.size())
    Q[S<eps]=0
    Q=Q.diag_embed()
    g=sym(grad_output_st)
    S[S<eps]=eps
    dLdu= 2* g @ u @ S.diag_embed()
    dLdS= Q @ u.transpose(-1,-2) @ g @ u
    idx=torch.arange(0,dLdS.size(3), out=torch.LongTensor())
    k=dLdS[:,:,:,idx,idx].diag_embed()
    grad_input_st=u @ (( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    return grad_input_st,None
class ReEig_st(nn.Module):
  def __init__(self,eps=10**(-4)):
    super(ReEig_st,self).__init__()
    self.eps=eps

  def forward(self,input_st):
    return ReEig_st_function.apply(input_st,self.eps)
