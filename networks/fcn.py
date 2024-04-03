

from layers import *

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
class FCN(nn.Module):
    
    def __init__(self, output_size = 1):
        super(FCN, self).__init__()

        self.flatten = Flatten()
        self.ouput_size = output_size
        
        
    def forward(self, x):
        x = nn.Conv2d(x.shape[1], x.shape[1]//2, 3, padding = 'valid').to('cuda')(x)
        x = nn.BatchNorm2d(x.shape[1]).to('cuda')(x)
        x = nn.ReLU().to('cuda')(x)
        
        x = nn.Conv2d(x.shape[1], x.shape[1]//2, 3, padding = 'valid').to('cuda')(x)
        x = nn.BatchNorm2d(x.shape[1]).to('cuda')(x)
        x = nn.ReLU().to('cuda')(x)
        
        x = nn.Conv2d(x.shape[1], x.shape[1]//2, 3, padding = 'valid').to('cuda')(x)
        x = nn.BatchNorm2d(x.shape[1]).to('cuda')(x)
        x = nn.ReLU().to('cuda')(x)
        
        x = self.flatten(x)
        x = nn.Linear(x.shape[-1], x.shape[-1]//2).to('cuda')(x)
        x = nn.ReLU().to('cuda')(x)
        
        x = nn.Linear(x.shape[-1], x.shape[-1]//2).to('cuda')(x)
        x = nn.ReLU().to('cuda')(x)
        
        x = nn.Linear(x.shape[-1], 1).to('cuda')(x)
        x = nn.Linear(x.shape[0], self.ouput_size).to('cuda')(x.transpose(0,1))
        x = nn.Sigmoid().to('cuda')(x)
        
        return x
    