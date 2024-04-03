from __future__ import absolute_import, division, print_function

from layers import *

# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.utils import save_image

class GaussianLayer(nn.Module):
    def __init__(self, input_size):
        super(GaussianLayer, self).__init__()
        
        self.input_size = input_size
        # self.sigma = nn.Parameter(torch.rand(1))
        # self.gauss_layer  = gkern(input_size, self.sigma)
   
    def gaussian_fn(self, M, std):
        n = torch.arange(0, M) - (M - 1.0) / 2.0
        sig2 = 2 * std * std
        n = n.to('cuda')
        sig2.to('cuda')
        w = torch.exp(-n ** 2 / sig2).to('cuda')
        return w

    def gkern(self, kernlen=256, std=0.5):
        """Returns a 2D Gaussian kernel array."""
        std = std*255
        gkern1d = self.gaussian_fn(kernlen, std=std) 
        gkern2d = torch.outer(gkern1d, gkern1d)
        gkern2d = gkern2d[None, :, :]
        gkern2d = gkern2d.expand(3, kernlen, kernlen)
        return gkern2d
        
    def forward(self, sigmas):
        output  = [self.gkern(self.input_size, sigmas[i])[None, :, :, :] for i in range(sigmas.shape[0])]     
        return torch.concat(output)

    # def weights_init(self):
    #     n= np.zeros((21,21))
    #     n[10,10] = 1
    #     k = scipy.ndimage.gaussian_filter(n,sigma=3)
    #     for name, f in self.named_parameters():
    #         f.data.copy_(torch.from_numpy(k))
            



# p = gkern(192, torch.rand(1))
# save_image(p, 'gaussian_check.png')
# print('cmpl')

# A = np.random.rand(256*256).reshape([256,256])
# A = torch.from_numpy(A)
# guassian_filter = gkern(256, std=32)


# class GaussianLayer(nn.Module):
#     def __init__(self):
#         super(GaussianLayer, self).__init__()
#         self.seq = nn.Sequential(
#             nn.ReflectionPad2d(10), 
#             nn.Conv2d(3, 3, 21, stride=1, padding=0, bias=None, groups=3)
#         )

#         self.weights_init()
        
#     def forward(self, x):
#         return self.seq(x)

#     def weights_init(self):
#         n= np.zeros((21,21))
#         n[10,10] = 1
#         k = scipy.ndimage.gaussian_filter(n,sigma=3)
#         for name, f in self.named_parameters():
#             f.data.copy_(torch.from_numpy(k))
            