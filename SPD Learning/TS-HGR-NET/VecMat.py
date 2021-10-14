class VecMat_ts_function(Function):

  @staticmethod
  def forward(ctx,input_ts):
    ctx.save_for_backward(input_ts)
    #TS
    row=input_ts.size(-1)
    input_ts.abs_()
    input_ts+=(sqrt(2)-1)*input_ts.triu(1)
    id=torch.LongTensor([[i,j] for i in range(row) for j in range(i,row)]).T
    output_ts=input_ts[:,:,:,:,:,id[0],id[1]].unsqueeze(-1)
    return output_ts

  @staticmethod
  def backward(ctx,grad_output_ts):
    input_ts=ctx.saved_tensors
    input_ts=input_ts[0]
    #TS
    batch,seq,fingers,NS,joints,row,col=input_ts.size()
    grad_input_ts=torch.zeros(input_ts.size())
    j=0
    for i in range(row):
      grad_input_ts[:,:,:,:,:,i,i:]=grad_output_ts[:,:,:,:,:,j:j+row-i,0]
      grad_input_ts[:,:,:,:,:,i:,i]=grad_input_ts[:,:,:,:,:,i,i:]
      j+=row-i
    grad_input_ts+=(sqrt(2)-1)*(grad_input_ts.triu(1)+grad_input_ts.tril(-1))
    return grad_input_ts
class VecMat_ts(nn.Module):
  def __init__(self):
    super(VecMat_ts,self).__init__()

  def forward(self,input_ts):
    return VecMat_ts_function.apply(input_ts)
