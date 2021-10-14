

class Gauss_agg1_st_function(Function):
  @staticmethod
  def forward(ctx,input,t0):
    
    t0=torch.tensor(t0)
    ctx.save_for_backward(input,t0)
    batch,nb_frames,joints,coor,col=input.size()
    #ST
    output_st=[]
    for s in range(6):
      binf,bsup=min(sequence(nb_frames)[s]),max(sequence(nb_frames)[s])
      x=input[:,binf:bsup+1].clone().transpose(1,2).reshape(batch,5,4,bsup-binf+1,coor,col)
      y=x.clone()
      x[:,:,1:-1]=(x[:,:,1:-1]+x[:,:,:-2]+x[:,:,2:])/3
      mu=x.mean(2)
      cov=torch.zeros(batch,5,bsup-binf+1,coor,coor)
      m=mu.unsqueeze(2).expand(x.size())
      xm,x0,xp=y[:,:,:,:-2]-m[:,:,:,1:-1], y-m, y[:,:,:,2:]-m[:,:,:,1:-1]
      cov[:,:,1:-1]=((xm @ xm.transpose(-1,-2) +x0[:,:,:,1:-1] @ x0[:,:,:,1:-1].transpose(-1,-2) + xp @ xp.transpose(-1,-2))/3).mean(2)
      cov[:,:,::bsup-binf]=(x0[:,:,:,::bsup-binf] @ x0[:,:,:,::bsup-binf].transpose(-1,-2)).mean(2)
  
      elt00 = cov + mu @ mu.transpose(-1,-2)
      elt01 = mu
      elt10 = mu.transpose(-1,-2)
      elt11 = torch.ones(batch,5,bsup-binf+1,1,1)
      output_st.append(torch.cat((torch.cat((elt00,elt01),-1),torch.cat((elt10,elt11),-1)),-2))
    return torch.cat(tuple(output_st),2)


  @staticmethod
  def backward(ctx,grad_output_st):
    
    input,t0 = ctx.saved_tensors
    t0=int(t0)
    batch,nb_frames,joints,coor,col=input.size()
    grad_input_st=torch.zeros(input.size())
    grad_output_st=grad_output_st.split([len(sequence(nb_frames)[s]) for s in range(6)],2)
    input=input.reshape(batch,nb_frames,joints,coor)
    for s in range(6):
      g=sym(grad_output_st[s]).transpose(1,2)
      binf,bsup=min(sequence(nb_frames)[s]),max(sequence(nb_frames)[s])
      X=input[:,binf:bsup+1].clone().reshape(batch,bsup-binf+1,5,4,coor)
      #outside the edges of frames
      Xs=torch.cat((X[:,1:-1],X[:,:-2],X[:,2:]),-2)
      B=torch.eye(coor+1,coor).reshape(1,1,1,coor+1,coor).expand(batch,bsup-binf-1,5,coor+1,coor)
      b=torch.cat((torch.zeros(coor),torch.ones(1))).reshape(1,1,1,coor+1,1).expand(batch,bsup-binf-1,5,coor+1,1)
      vect_one=torch.ones(batch,bsup-binf-1,5,12,1)
      x=(1/6)*(Xs @ B.transpose(-1,-2) + vect_one @ b.transpose(-1,-2) ) @ g[:,1:-1] @ B
      grad_input_st[:,binf+1:bsup]+=((x[:,:,:,:4]+x[:,:,:,4:8]+x[:,:,:,8:12])/3).reshape(batch,bsup-binf-1,20,coor).unsqueeze(-1)
      #The edges of frames
      Xs=X[:,::bsup-binf].reshape(batch,2,5,4,coor)
      B=torch.eye(coor+1,coor).reshape(1,1,1,coor+1,coor).expand(batch,2,5,coor+1,coor)
      b=torch.cat((torch.zeros(coor),torch.ones(1))).reshape(1,1,1,coor+1,1).expand(batch,2,5,coor+1,1)
      vect_one=torch.ones(batch,2,5,4,1)
      x=(1/2)*(Xs @ B.transpose(-1,-2) + vect_one @ b.transpose(-1,-2) ) @ g[:,::bsup-binf] @ B
      grad_input_st[:,binf:bsup+1:bsup-binf]+=x.reshape(batch,2,20,coor).unsqueeze(-1)    
    return grad_input_st/3,None
class Gauss_agg1_st(nn.Module):
  def __init__(self,t0=1):
    super(Gauss_agg1_st,self).__init__()
    self.t0=t0
  def forward(self,input):
    return Gauss_agg1_st_function.apply(input,self.t0)
