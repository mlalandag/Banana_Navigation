import torch

class SNN(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape):
        
        print("init SNN")
        
        super(SNN, self).__init__()
        
        self.linear1 = torch.nn.Linear(input_shape, 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.out = torch.nn.Linear(64, output_shape)

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.out(x)
        
        return x