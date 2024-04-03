

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

import wandb_logging
import wandb
import numpy as np

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

# from datasets import SCAREDRAWDataset

from datasets import LungRAWDataset

from torchvision.utils import save_image, make_grid

import wandb_logging

import torchvision.transforms as transforms

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# losses 
def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    losses = {}
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1).mean(-1).mean(-1)
    

    ssim_loss = ssim(pred, target).mean(1).mean(-1).mean(-1)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    
    losses['l1'] = l1_loss.mean(-1)
    losses['ssim_loss'] = ssim_loss.mean(-1)
    losses['reprojection_loss'] = reprojection_loss.mean(-1)
    
    return losses

def save_model(epoch, log_path, models, model_optimizer):
    """Save model weights to disk
    """
    save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()
        if model_name == 'encoder':
            # save the sizes - these are needed at prediction time
            to_save['height'] = height
            to_save['width'] = width
        torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(model_optimizer.state_dict(), save_path)

def load_model_fxn(load_weights_folder, models_to_load, models):
    """Load model(s) from disk
    """
    load_weights_folder = os.path.expanduser(load_weights_folder)

    assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
    print("loading model from folder {}".format(load_weights_folder))

    for n in models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_weights_folder, "{}.pth".format(n))
        model_dict = models[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[n].load_state_dict(model_dict)

    # loading adam state
    # optimizer_load_path = os.path.join(load_weights_folder, "adam.pth")
    # if os.path.isfile(optimizer_load_path):
    #     print("Loading Adam weights")
    #     optimizer_dict = torch.load(optimizer_load_path)
    #     model_optimizer.load_state_dict(optimizer_dict)
    # else:
    #     print("Cannot find Adam weights so Adam is randomly initialized")
            
# model 
# optimizer 
# wandbloggig 
# train
# save model
# save optim 
# load model 
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
load_model = True 
# data loader
models = {}
input_size = ()
learning_rate = 10e-06
batch_size = 16
num_epochs = 20
parameters_to_train = []

# variables
data_path         = os.path.join(file_dir, "data")
height            = 192
width             = 192
frame_ids         = [0, -1, 1]
adversarial_prior = False
device            = torch.device("cuda")
n_channels        = 3
n_classes         = 3

# wandb 
config = dict(
    height = height,
    width = width,
    epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    dataset="only_unet_training",
    frame_ids = frame_ids,
    augmentation = "True",
    align_corner="True")
        
wandb.login()
wandb.init(project="gaussian_test", config= config, dir = 'data/logs')

models['decompose'] = networks.UNet(n_channels, n_classes) 
models['gaussian'] = networks.GaussianLayer(height) 
models['sigma'] = networks.FCN(1024) 

if load_model:
    load_model_fxn('code/logs/models/weights_0', ["decompose"], models)
    



models['decompose'].to(device)
models['sigma'].to(device)
models['gaussian'].to(device)

parameters_to_train += list(models["decompose"].parameters())
parameters_to_train += list(models["gaussian"].parameters())
parameters_to_train += list(models["sigma"].parameters())

# optimizer
model_optimizer = optim.Adam(parameters_to_train, learning_rate)

# dataloader 
datasets_dict = {"endovis": datasets.LungRAWDataset}
dataset = datasets_dict['endovis']

fpath = os.path.join(os.path.dirname(__file__), "splits", "endovis", "{}_files_phantom.txt")

train_filenames = readlines(fpath.format("train"))
val_filenames = readlines(fpath.format("val"))
img_ext = '.png'

num_train_samples = len(train_filenames)
num_total_steps = num_train_samples // batch_size * num_epochs

train_dataset =  dataset(
     data_path, train_filenames,  height,  width,
     frame_ids, 4, is_train=True, img_ext=img_ext, adversarial_prior =  adversarial_prior, len_ct_depth_data = 2271)
train_loader = DataLoader(train_dataset, batch_size, True, drop_last=True)
    
val_dataset = dataset(data_path, val_filenames,  height,  width,frame_ids, 4, is_train=False, img_ext=img_ext, adversarial_prior = False, len_ct_depth_data = 0) 
val_loader = DataLoader(val_dataset,  batch_size, True, drop_last=True)
val_iter = iter( val_loader)




models['decompose'].to(device)
models['gaussian'].to(device)
models['sigma'].to(device)

# wandb model watch 
# for key, model in models.items():
# wandb.watch(models['decompose'], log_freq=1, log='all') # default is 1000, it makes the model very slow

# losses 
ssim = SSIM()
ssim.to(device)


# train loop
epoch = 0
step = 0
start_time = time.time()
step = 1
save_frequency = 50
custom_step = 0
for  epoch in range(num_epochs):
    print("Training")
    models['decompose'].train()
    # models['gaussian'].train()
    # models['sigma'].train()
    
    custom_step+=1
    outputs = {}
    for batch_idx, inputs in enumerate(train_loader):

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(device)
            
        before_op_time = time.time()

        features                = models["decompose"](inputs["color_aug", 0, 0])
        outputs['decompose']    = features[1]
        sigma_out               = models['sigma'](features[0]) # check the dimension. pass through the convolutions. Input in the gaussian network 
        gaussian_mask           = models["gaussian"](sigma_out)
        
        outputs['compose'] = outputs['decompose'] * gaussian_mask.repeat(16,1,1,1)
        # outputs['compose'] = outputs['decompose']
        
        
        losses = compute_reprojection_loss(outputs['compose'], inputs["color_aug", 0, 0])
        
        model_optimizer.zero_grad()
        losses['reprojection_loss'].backward()
        model_optimizer.step()

        duration = time.time() - before_op_time
        
        step+=1
        
        # wand_b loggin 
        if ( step + 1) %  save_frequency == 0:
            
            
            models['decompose'].eval()
            models['sigma'].eval()
            models['gaussian'].eval()
            
            # features_val = models["decompose"](inputs["color_aug", 0, 0][0][None, :, :, :])
            features_val = models["decompose"](inputs["color_aug", 0, 0])
            image        = features_val[1]
            sigma_out_val               = models['sigma'](features_val[0]) # check the dimension. pass through the convolutions. Input in the gaussian network 
            gaussian_mask_val           = models["gaussian"](sigma_out_val)
            final_val = image*gaussian_mask_val.repeat(16,1,1,1)
            
            wandb.log({"{}".format('train_gaussmask'):wandb.Image(make_grid(gaussian_mask_val), caption = 'gaussian mask'),'custom_step':custom_step})  
            wandb.log({"{}".format('train_original'):wandb.Image(inputs["color_aug", 0, 0], caption ='original image'),'custom_step':custom_step})  
            wandb.log({"{}".format('train_reconstructed'):wandb.Image(make_grid(final_val), caption = 'reconstructed image'),'custom_step':custom_step})  
            wandb.log({"{}".format('train_intermediate'):wandb.Image(make_grid(image), caption = 'intermediate image'),'custom_step':custom_step})  
            
            for l, v in losses.items():
                wandb.log({"{}_{}".format('train', l):v, 'custom_step':custom_step})
    
    #save model
    # save_model(epoch, 'code/logs', models, model_optimizer)
    
        
    
    
