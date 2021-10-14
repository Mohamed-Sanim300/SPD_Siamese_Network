class Net(nn.Module):
    def __init__(self):
        self.tangent = SPDTangentSpace(200)
        self.linear = nn.Linear(20100, 14, bias=True)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.tangent(x)
        x = self.linear(x.type(torch.FloatTensor))
        return x

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
