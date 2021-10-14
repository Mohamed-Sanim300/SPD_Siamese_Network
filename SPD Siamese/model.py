class Net(nn.Module):
    def __init__(self):
        self.tangent = SPDTangentSpace(200)
        self.linear = nn.Linear(20100, 14, bias=True)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.trans1(x)
        x = self.rect1(x)
        x = self.tangent(x)
        # x = self.dropout(x)
        x = self.linear(x.type(torch.FloatTensor))
        return x
