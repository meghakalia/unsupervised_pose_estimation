import torch.nn as nn
from layers import *

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        # self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)
        self.base_filters = 16
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, self.base_filters, normalize=False),
            *discriminator_block(self.base_filters, self.base_filters*2),
            *discriminator_block(self.base_filters*2, self.base_filters*4),
            # *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(self.base_filters*4, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
    
    
    
class DiscriminatorUnet(nn.Module):
    def __init__(self, input_shape):
        super(DiscriminatorUnet, self).__init__()

        channels, height, width = input_shape

        self.output_shape = input_shape
        
        # Calculate output shape of image discriminator (PatchGAN)
        # self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        
        
        
        self.bilinear = True

        self.base_filter = 8
        self.inc = (DoubleConv(channels, self.base_filter))
        self.down1 = (Down(self.base_filter, self.base_filter*2)) # self.base_filter, self.base_filter*2
        self.down2 = (Down(self.base_filter*2, self.base_filter*4))# self.base_filter*2, self.base_filter*4
        self.down3 = (Down(self.base_filter*4, self.base_filter*8))# self.base_filter*4, # self.base_filter*8
        factor = 2 if self.bilinear else 1
        self.down4 = (Down(self.base_filter*8, self.base_filter*16 // factor)) # self.base_filter*8, self.base_filter*16
        self.up1 = (Up(self.base_filter*16, self.base_filter*8 // factor, self.bilinear))# self.base_filter*16, self.base_filter*8
        self.up2 = (Up(self.base_filter*8, self.base_filter*4 // factor, self.bilinear))# self.base_filter*8, self.base_filter*4
        self.up3 = (Up(self.base_filter*4, self.base_filter*2// factor, self.bilinear))# self.base_filter*4, self.base_filter*2
        self.up4 = (Up(self.base_filter*2, self.base_filter*1, self.bilinear)) # self.base_filter*2, self.base_filter*1
        self.outc = (OutConv(self.base_filter, 1))
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
    
        # output = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # output.append(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # output.append(self.sigmoid(logits))
        return logits

    # def forward(self, img):
    #     return self.model(img)