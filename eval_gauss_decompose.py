
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

#pip install torchmetrics[image]
from torchmetrics.image.fid import FrechetInceptionDistance

fid_criterion = FrechetInceptionDistance(feature = 64, normalize=True).to('cuda')

model_folders = ['/code/code/4_multigaussian_gauss_sum_2_singleGaussaNetwork_recon_pretrained_trainable_dataaug_True_gauss_num_1_batchnorm_True_ssim_l1_0.9500000000000001_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_14',
                 '/code/code/4_multigaussian_gauss_sum_2_singleGaussaNetwork_recon_pretrained_trainable_dataaug_True_gauss_num_1_batchnorm_True_ssim_l1_0.8500000000000001_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_4',
                 '/code/code/4_multigaussian_gauss_sum_2_singleGaussaNetwork_recon_pretrained_trainable_dataaug_True_gauss_num_1_batchnorm_True_ssim_l1_0.75_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_7',
                 '/code/code/4_multigaussian_gauss_sum_2_singleGaussaNetwork_recon_pretrained_trainable_dataaug_True_gauss_num_1_batchnorm_True_ssim_l1_0.65_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_20',
                 '/code/code/4_multigaussian_gauss_sum_2_singleGaussaNetwork_recon_pretrained_trainable_dataaug_False_gauss_num_1_batchnorm_True_ssim_l1_0.8500000000000001_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_13',
                 '/code/code/4_multigaussian_gauss_sum_2_singleGaussaNetwork_recon_pretrained_trainable_dataaug_False_gauss_num_1_batchnorm_True_ssim_l1_0.75_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_18',
                 '/code/code/4_multigaussian_gauss_sum_2_singleGaussaNetwork_recon_pretrained_trainable_dataaug_False_gauss_num_1_batchnorm_True_ssim_l1_0.65_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_15'
                 ]

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
        
def evaluation_FID(metric, imgs_dist1, imgs_dist2):
    metric.update(imgs_dist1, real=True)
    metric.update(imgs_dist2, real=False)
    return metric.compute()

def set_eval(models):
    models['decompose'].eval()
    models['sigma_combined'].eval()
    models['gaussian1'].eval()

# define models
# load the model      
# run on dataset validation
# print the FID score with the model name 

# variales 
 # frac = 0.
 
for model_path in model_folders:
    file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
    # data loader
    models = {}
    input_size = ()
    batch_size = 8
    # variables
    data_path         = os.path.join(file_dir, "data")
    height            = 192
    width             = 192
    frame_ids         = [0, -1, 1]
    adversarial_prior = False
    device            = torch.device("cuda")
    n_channels        = 3
    n_classes         = 3
    scheduler_step_size = 8
    bool_multi_gauss = True
    separate_mean_std = True
    same_gauss_kernel = False # if all images have same gaussian profile
    batch_norm = True 
    data_aug = False
    # gauss_number =  1 + np.random.randint(2, 5, size= 1)
    gauss_number =  1

    load_weights_folder = ''
    models_to_load = ''

    models['decompose'] = networks.UNet(n_channels, n_classes)
    models['sigma_combined'] = networks.FCN(output_size = 16) # 4 sigma and mu x 3 for 3 gaussians
    models['gaussian{}'.format(1)] = networks.GaussianLayer(height)

    models['decompose'].to(device)
    models['sigma_combined'].to(device)
    models['gaussian{}'.format(1)].to(device)

    load_model_fxn(model_path, ["decompose", "sigma_combined", "gaussian1"], models)

    # dataloader 
    datasets_dict = {"endovis": datasets.LungRAWDataset}
    dataset = datasets_dict['endovis']

    fpath = os.path.join(os.path.dirname(__file__), "splits", "endovis", "{}_files_phantom.txt")

    val_filenames = readlines(fpath.format("val"))[:500]
    img_ext = '.png'

    val_dataset = dataset(data_path, val_filenames,  height,  width,frame_ids, 4, is_train=False, img_ext=img_ext, adversarial_prior = False, len_ct_depth_data = 0,  data_augment = data_aug) 
    val_loader = DataLoader(val_dataset,  batch_size, shuffle = True, drop_last=True)
    val_iter = iter( val_loader)

    # evauation metric (FID, SSIM)

    ssim = SSIM()
    ssim.to(device)
    ssim.eval()

    # prediction 
    # set the models to eval 

    total_fid = 0
    total_ssim = 0     

    batch_count = 1
    for batch_idx, inputs in enumerate(val_loader):
        set_eval(models)
        
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(device)
           
        features   = models["decompose"](inputs["color", 0, 0]) # no augmentation for validation 
        image    = features[1]
        
        sigma_out_combined        = models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
        gaussian_mask1            = models["gaussian1"](sigma_out_combined[:, :4])
        gaussian_mask2            = models["gaussian1"](sigma_out_combined[:, 4:8])
        gaussian_mask3            = models["gaussian1"](sigma_out_combined[:, 8:12])
        gaussian_mask4            = models["gaussian1"](sigma_out_combined[:, 12:16])
        
        image_recon = image * (gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)
        
        total_fid+=evaluation_FID(fid_criterion, inputs["color", 0, 0], image_recon)
        # total_ssim+=ssim(inputs["color", 0, 0], image_recon).mean(1).mean(-1).mean(-1).mean()
        batch_count+=1

    # total evaluation 
    total_fid = total_fid/batch_count
    total_ssim= total_ssim /batch_count

    print('total_fid {} total_fid {} model_name {}'.format(total_fid, total_ssim, model_path))