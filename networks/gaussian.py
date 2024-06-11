from __future__ import absolute_import, division, print_function

from layers import *

# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.utils import save_image

class GaussianLayer(nn.Module):
    def __init__(self, input_size, num_of_gaussians = 1):
        super(GaussianLayer, self).__init__()
        
        self.input_size = input_size
        self.num_of_gaussians = num_of_gaussians
        self.proportions_rand = torch.rand(self.num_of_gaussians)
        self.proportions = nn.Parameter(self.proportions_rand/torch.sum(self.proportions_rand))
        # self.sigma = nn.Parameter(torch.rand(1))
        # self.gauss_layer  = gkern(input_size, self.sigma)
        
        # self.sigmax = nn.Parameter(torch.rand(1))
        # self.sigmay = nn.Parameter(torch.rand(1))
        # self.meanx = nn.Parameter(torch.rand(1))
        # self.meany = nn.Parameter(torch.rand(1))
        
   
    def gaussian_fn(self, M, std, mean):
        #mean 0 - 1
    
        # n = torch.arange(0, M) - (M - 1.0) / 2.0
        n = torch.arange(0, M).to('cuda') - mean
        sig2 = 2 * std * std
        n = n.to('cuda')
        sig2.to('cuda')
        w = torch.exp(-n ** 2 / (sig2 + 1e-7)).to('cuda')
        return w

    def gkern(self, kernlen=256, stdx=0.5, stdy=0.5, meanx= 0.5, meany= 0.5):
        """Returns a 2D Gaussian kernel array."""
        stdx = stdx*kernlen
        stdy = stdy*kernlen
        meanx = meanx*kernlen
        meany = meany*kernlen
        gkern1d_x = self.gaussian_fn(kernlen, std=stdx, mean = meanx) 
        gkern1d_y = self.gaussian_fn(kernlen, std=stdy, mean = meany)
        gkern2d = torch.outer(gkern1d_x, gkern1d_y)
        gkern2d = gkern2d[None, :, :]
        # gkern2d = gkern2d.expand(3, kernlen, kernlen)
        return gkern2d
    
    def combine_gaussians(self):
        
        return 
    
    def forward(self, sigmas):
        final_out = []
        if self.num_of_gaussians==1:
            # output  = [self.gkern(self.input_size, sigmas[i])[None, :, :, :] for i in range(sigmas.shape[0])]  
            output  = [self.gkern(self.input_size, sigmas[i][0], sigmas[i][1], sigmas[i][2], sigmas[i][3])[None, :, :, :] for i in range(sigmas.shape[0])]
        else:
            # untested, if gaussian are more than 1 
            output = []
            for i in range(sigmas.shape[0]):             
                output_temp = self.proportions[0]*self.gkern(self.input_size, sigmas[i][0])
                for j in range(1, sigmas.shape[-1]):
                    output_temp += self.proportions[j]*self.gkern(self.input_size, sigmas[i][j])
                
                output+= [output_temp[None, :, :, :]]

        final_out.append(torch.concat(output))
        # final_out.append(self.proportions)
        return final_out

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
            