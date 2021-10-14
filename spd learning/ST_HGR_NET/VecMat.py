class VecMat_st_function(Function):

  @staticmethod
  def forward(ctx,input_st):
    ctx.save_for_backward(input_st)
    #ST
    batch,fingers,nb_frames,row,col=input_st.size()
    input_st.abs_()
    input_st+=(sqrt(2)-1)*input_st.triu(1)
    id=torch.LongTensor([[i,j] for i in range(row) for j in range(i,row)]).T
    output_st=input_st[:,:,:,id[0],id[1]].unsqueeze(-1)
    return output_st

  @staticmethod
  def backward(ctx,grad_output_st):
    input_st=ctx.saved_tensors
    input_st=input_st[0]
    #ST
    batch,fingers,nb_frames,row,col=input_st.size()
    g=torch.zeros(batch,fingers,nb_frames,row,col)
    j=0
    for i in range(row):
      g[:,:,:,i,i:]=grad_output_st[:,:,:,j:j+row-i,0]
      g[:,:,:,i:,i]=g[:,:,:,i,i:]
      j+=row-i
    g+=(sqrt(2)-1)*(g.triu(1)+g.tril(-1))
    return g
class VecMat_st(nn.Module):
  def __init__(self):
    super(VecMat_st,self).__init__()

  def forward(self,input_st):
    return VecMat_st_function.apply(input_st)
