import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size:list, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.fc5 = nn.Linear(hidden_size[3], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)
        return out

if __name__=="__main__":
    import torch

    model_1 = MLP(7, [64, 128, 256, 128], num_classes=7)
    model_2=MLP(90,[64,128,256,128],num_classes=7)
    batch_size=1024
    x1 = torch.randn((batch_size, 7))
    x2 = torch.randn((batch_size, 90))
    print(model_1)
    print("done")
    from thop import profile
    flops, params = profile(model_1, inputs=(x1,))
    flops_2, params_2 = profile(model_2, inputs=(x2,))
    print((flops+flops_2) / 1e9, (params+params_2) / 1e6)  # flops单位G，para单位M