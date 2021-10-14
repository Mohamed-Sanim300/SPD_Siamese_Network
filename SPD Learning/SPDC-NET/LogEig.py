class LogEig_spdc_function(torch.autograd.Function):
  @staticmethod
  def forward(ctx,input,vect):
    u,S,v=input.svd()
    ctx.save_for_backward(u,S,torch.tensor(vect))
    output=u @ S.log().diag_embed() @ u.transpose(-1,-2)
    if vect:
      row=output.size(-1)
      output.abs_()
      output+=(sqrt(2)-1)*output.triu(1)
      id=torch.LongTensor([[i,j] for i in range(row) for j in range(i,row)]).T
      output=output[:,id[0],id[1]]
    return output

  @staticmethod
  def backward(ctx,grad_output):
    u,S,vect= ctx.saved_tensors
    if vect:
      row=u.size(-2)
      grad_input=torch.zeros(u.size())
      j=0
      for i in range(row):
        grad_input[:,i,i:]=grad_output[:,j:j+row-i]
        grad_input[:,i:,i]=grad_input[:,i,i:]
        j+=row-i
      grad_input+=(sqrt(2)-1)*(grad_input.triu(1)+grad_input.tril(-1))
      grad_output=grad_input
    g=sym(grad_output)
    P = S.unsqueeze(-1).expand(u.size())
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    dLdu= 2* g @ u @ S.log().diag_embed()
    dLdS= (1/S).diag_embed()@ u.transpose(-1,-2) @ g @ u
    idx=torch.arange(0,dLdS.size(-1), out=torch.LongTensor())
    k=dLdS[:,idx,idx].diag_embed()
    grad_input=u @(( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    return grad_input,None
class LogEig_spdc(nn.Module):
  def __init__(self,vect=True):
    super(LogEig_spdc,self).__init__()
    self.vect=vect
  def forward(self,input):
    return LogEig_spdc_function.apply(input,self.vect)
