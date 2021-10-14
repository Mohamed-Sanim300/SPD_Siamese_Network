class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of 
        Stiefel manifold.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return self.data.__repr__()
class SPDAgg_function(torch.autograd.Function):
  @staticmethod
  def forward(ctx,input,weights):
    ctx.save_for_backward(input,weights)
    output=torch.sum(weights @ input @ (weights.transpose(-1,-2)) ,1 )
    return output

  @staticmethod
  def backward(ctx,grad_output):
    input,weight=ctx.saved_tensors
    g=grad_output.unsqueeze(1).expand(input.size(0),60,200,200)
    grad_input=weight.transpose(-1,-2) @ g @ weight
    grad_weight= 2* g @ weight @ input
    return grad_input,grad_weight
class SPD_Agg(nn.Module):
  def __init__(self,input_size=56,output_size=200):
    super(SPD_Agg,self).__init__()
    self.output_size=output_size
    self.input_size=input_size
    self.weight = StiefelParameter(torch.FloatTensor(60,output_size,input_size), requires_grad=True)
    nn.init.orthogonal_(self.weight).requires_grad_()
        
  def forward(self,input):
    weight=self.weight.expand(input.size(0),60,self.output_size,self.input_size)
    return SPDAgg_function.apply(input,weight)
