class Gauss_agg1_ts_function(Function):
  @staticmethod
  def forward(ctx,input_ts,NS):
    NS=torch.tensor(NS)
    ctx.save_for_backward(input_ts,NS)
    batch,nb_frames,joints,coordinates,col=input_ts.size()
    #TRY TS
    inputs=input_ts.reshape(batch,nb_frames,5,4,coordinates,col)
    mu=torch.zeros((batch,6,5,NS,4,coordinates,1))
    cov=torch.zeros((batch,6,5,NS,4,coordinates,coordinates))
    for s in range(6):
      binf,bsup=min(sequence(nb_frames)[s]),max(sequence(nb_frames)[s])
      nb_fr=int((bsup-binf+1)/NS)
      for k in range(NS-1):
        mu[:,s,:,k]=inputs[:,k*nb_fr:(k+1)*nb_fr].mean(1)
        x=inputs[:,k*nb_fr:(k+1)*nb_fr]-mu[:,s,:,k].unsqueeze(1).expand(batch,nb_fr,5,4,coordinates,1)
        cov[:,s,:,k]=(x @ x.transpose(-1,-2)).mean(1)
      k=NS-1
      mu[:,s,:,k]=inputs[:,k*nb_fr:].mean(1)
      x=inputs[:,k*nb_fr:nb_frames]-mu[:,s,:,k].unsqueeze(1).expand(inputs[:,k*nb_fr:nb_frames].size())
      cov[:,s,:,k]=(x @ x.transpose(-1,-2)).mean(1)
    elt00 = cov + mu @ mu.transpose(-1,-2)
    elt01 = mu
    elt10 = mu.transpose(-1,-2)
    elt11 = torch.ones(batch,6,5,NS,4,1,1)
    return torch.cat((torch.cat((elt00,elt01),-1),torch.cat((elt10,elt11),-1)),-2)

  @staticmethod
  def backward(ctx,grad_output_ts):
    input_ts,NS = ctx.saved_tensors
    NS=int(NS)
    batch,nb_frames,joints,row,col=input_ts.size()
    grad_input_ts=torch.zeros(input_ts.size())
    inputs=input_ts.transpose(1,2).squeeze().reshape(batch,5,4,nb_frames,row).type(torch.DoubleTensor)
    #TS
    g=sym(grad_output_ts).type(torch.DoubleTensor)
    B=torch.eye(row+1,row).reshape(1,1,1,row+1,row).expand(batch,5,4,row+1,row).type(torch.DoubleTensor)
    b=torch.cat((torch.zeros(row),torch.ones(1))).reshape(1,1,1,1,row+1).expand(batch,5,4,1,row+1)
    for s in range(6):
      binf,bsup=min(sequence(nb_frames)[s]),max(sequence(nb_frames)[s])
      nb_fr=int((bsup-binf+1)/NS)
      vect_one=torch.ones(batch,5,4,nb_fr,1)
      for k in range(NS-1):
        x = (2/nb_fr)* (inputs[:,:,:,k*nb_fr:(k+1)*nb_fr] @ B.transpose(-1,-2) + vect_one @ b) @ g[:,s,:,k] @ B
        grad_input_ts[:,k*nb_fr:(k+1)*nb_fr]+=x.reshape(batch,20,nb_fr,row,col).transpose(1,2)
      k=NS-1
      rest_fr=inputs[0,0,0,k*nb_fr:].size(0)
      vect_one=torch.ones(batch,5,4,rest_fr,1)
      x = (2/nb_fr)* (inputs[:,:,:,k*nb_fr:] @ B.transpose(-1,-2) + vect_one @ b) @ g[:,s,:,k] @ B
      grad_input_ts[:,k*nb_fr:]+=x.reshape(batch,20,rest_fr,row,col).transpose(1,2)
    return grad_input_ts/3,None
  
class Gauss_agg1_ts(nn.Module):
  def __init__(self,NS=15):
    super(Gauss_agg1_ts,self).__init__()
    self.NS=NS
  def forward(self,input):
    return Gauss_agg1_ts_function.apply(input,self.NS)
