

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
        self.dropout = nn.Dropout(0.2).to('cuda')
        self.nonlinear = nn.LeakyReLU(negative_slope=0.01).to('cuda')
        
        self.normalization = []
        self.conv = []
        self.linear = []
        
        base_channels = 512
        for i in range(1, 4):
            self.normalization.append(nn.InstanceNorm2d(base_channels//2**(i)).to('cuda'))
            self.conv.append(nn.Conv2d(base_channels//2**(i-1), base_channels//2**(i), 3, padding = 'valid').to('cuda'))
        
        ch = 64*6*6
        final_ch_size = 0 
        for i in range(1,3):
           self.linear.append(nn.Linear(ch//2**(i-1), ch//2**(i)).to('cuda'))
           final_ch_size = ch//2**(i)
        
        # the last one
        self.linear.append(nn.Linear(final_ch_size, output_size).to('cuda'))
        
        self.relu = nn.ReLU().to('cuda')
        
        # self.initializeWeights()
       
    
    # def initializeWeights(self):
    #     nn.init.constant_(self.relu, 0.5)
        
    #     return 
    
    def forward(self, x):
        x = self.conv[0](x)
        # x = nn.BatchNorm2d(x.shape[1]).to('cuda')(x)
        x = self.normalization[0](x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        
        x = self.conv[1](x)
        x = self.normalization[1](x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        
        x = self.conv[2](x)
        x = self.normalization[2](x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        
        x = self.flatten(x)
        x = self.linear[0](x)
        x = self.nonlinear(x)
        
        x = self.linear[1](x)
        x = self.nonlinear(x)
        
        x = self.linear[2](x)
        # x = nn.Linear(x.shape[0], self.ouput_size).to('cuda')(x.transpose(0,1))
        # x = self.relu(x)
        
        x = nn.Sigmoid().to('cuda')(x)
        
        return x
    