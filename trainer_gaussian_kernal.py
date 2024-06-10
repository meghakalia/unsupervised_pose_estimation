

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

from torchmetrics.image.fid import FrechetInceptionDistance

fid_criterion = FrechetInceptionDistance(feature = 64, normalize=True).to('cuda')

def compute_disc_loss(inputs, prediction):
    # backpropagate through discriminator
        Discriminator.train()
        optimizer_Discriminator.zero_grad()

        ct_loss_disc = Discriminator(inputs)
        loss_real = criterion_Discriminator(ct_loss_disc, valid)
        
        loss_fake = 0 

        depth_disc_res = Discriminator(prediction)
        loss_fake+=criterion_Discriminator(depth_disc_res, fake)
        
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        
        return loss_D
    
def discriminator_train_step(inputs, prediction):
        
        # real image 
        # outputs : Predicted 
        # for key, ipt in inputs.items():
        #     inputs[key] = ipt.to(device)
            
        # backpropagate through discriminator
        Discriminator.train()
        optimizer_Discriminator.zero_grad()

        ct_loss_disc = Discriminator(inputs)
        loss_real = criterion_Discriminator(ct_loss_disc, valid)
        
        loss_fake = 0 

        depth_disc_res = Discriminator(prediction)
        loss_fake+=criterion_Discriminator(depth_disc_res, fake)
            
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        
        optimizer_Discriminator.step()
        
        Discriminator.eval()
       
        return loss_D
    
# 0.25 - 0.95
# losses 
def compute_reprojection_loss(pred, target, frac = 0.45):
    """Computes reprojection loss between a batch of predicted and target images
    """
    losses = {}
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1).mean(-1).mean(-1)
    

    ssim_loss = ssim(pred, target).mean(1).mean(-1).mean(-1)
    reprojection_loss = (1.0 - frac)* ssim_loss + frac * l1_loss
    
    if discriminator:
        Discriminator.eval()
        output_disc = Discriminator(pred) # this may not be correct because of the scale. check whether we want to miniize disp or depth
        disc_loss = criterion_Discriminator(output_disc, valid)
        losses["discriminator"] = disc_loss
        

    # reprojection_loss = l1_loss
    
    losses['l1'] = l1_loss.mean(-1)
    # losses['ssim_loss'] = 0
    if discriminator:
        losses['reprojection_loss'] = reprojection_loss.mean(-1) + losses["discriminator"]
    else:
        losses['reprojection_loss'] = reprojection_loss.mean(-1) 
        
    losses['ssim_loss'] = ssim_loss.mean(-1)
    
    
        
    
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

def evaluation_FID(metric, imgs_dist1, imgs_dist2):
    metric.update(imgs_dist1, real=True)
    metric.update(imgs_dist2, real=False)
    return metric.compute()
    
    
    
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

for z in range(1, 5):
    frac = 0.55 + z*0.10
    
    # frac = 0.
    file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
    load_model = True 
    # data loader
    models = {}
    input_size = ()
    learning_rate = 10e-08 # original is 10e-06
    batch_size = 4
    num_epochs = 25
    parameters_to_train = []
    train_unet_only = False


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
    discriminator = True
    same_gauss_kernel = False # if all images have same gaussian profile
    batch_norm = True 
    data_aug = True
    # gauss_number =  1 + np.random.randint(2, 5, size= 1)
    gauss_number =  1
    
    free_mask = False
    
    discriminator_lr = 0.0002
    b1               = 0.5
    b2               = 0.999

    experiment_name = "discriminator_batch_4_dataaug_{}_gauss_num_{}_batchnorm_{}_ssim_l1_{}_sigma_network_gauss_combination{}_same_gausskernel_{}_separatemeanstd_{}".format(data_aug, gauss_number, batch_norm, frac, bool_multi_gauss, same_gauss_kernel, separate_mean_std)
    # wandb 
    config = dict(
        height = height,
        width = width,
        epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataset=experiment_name,
        frame_ids = frame_ids,
        augmentation = data_aug,
        align_corner="True", 
        pretrained_unet_model = load_model, 
        multi_gauss = bool_multi_gauss, 
        same_gauss_assumption = same_gauss_kernel)
            
    wandb.login()
    wandb.init(project="gaussian_test", config= config, dir = 'data/logs', name = experiment_name)

    # models['decompose'] = networks.UNet_instanceNorm(n_channels, n_classes) 
    if batch_norm: 
        models['decompose'] = networks.UNet(n_channels, n_classes) 
    else:
        models['decompose'] = networks.UNet_instanceNorm(n_channels, n_classes)

    if not train_unet_only: 
        
        if separate_mean_std: 
            models['sigma_combined'] = networks.FCN(output_size = 16) # 4 sigma and mu x 3 for 3 gaussians
            
            if free_mask:
                models['free_mask'] = networks.FCN_free_mask(output_size = 3) # 3 different masks for each of channels
            # models['gaussian'] = networks.GaussianLayer(height)
            
            for g in range(1, gauss_number+1): 
                models['sigma{}'.format(g)] = networks.FCN(output_size = 10) # 4 for each of std x, std y, mean x , mean y, two for the mixture of gaussians 
                models['gaussian{}'.format(g)] = networks.GaussianLayer(height)
        
        if discriminator:
            criterion_Discriminator = torch.nn.BCEWithLogitsLoss()
            
            input_shape = (3, width, height)
            # define a discriminator to give feedback on every patch. 
            Discriminator = networks.Discriminator(input_shape)
            Discriminator.to(device)
            
            # Adversarial ground truths
            #  valid = Variable(Tensor(np.ones((real_A.size(0), *D_A2.output_shape))), requires_grad=False)
            valid = torch.ones((1, *Discriminator.output_shape), requires_grad=False).repeat([batch_size, 1, 1, 1]).to(device)
            fake = torch.zeros((1, *Discriminator.output_shape), requires_grad=False).repeat([batch_size, 1, 1, 1]).to(device)
            
            optimizer_Discriminator = torch.optim.Adam(Discriminator.parameters(), lr=discriminator_lr, betas=(b1, b2))
            
            
    models['decompose'].to(device)
    models['sigma_combined'].to(device)
    if free_mask:
        models['free_mask'].to(device)
    
    # models['gaussian'].to(device)

    if not train_unet_only:
        for g in range(1, gauss_number+1): 
            models['sigma{}'.format(g)].to(device)
            models['gaussian{}'.format(g)].to(device)
            
        # models['sigma1'].to(device)
        # models['gaussian1'].to(device)
        
        # models['sigma2'].to(device)
        # models['gaussian2'].to(device)

    parameters_to_train += list(models["decompose"].parameters())
    parameters_to_train += list(models['sigma_combined'].parameters())
    if free_mask:
        parameters_to_train += list(models['free_mask'].parameters())
    # parameters_to_train += list(models['gaussian'].parameters())

    for g in range(1, gauss_number+1): 
        parameters_to_train += list(models['gaussian{}'.format(g)].parameters())
        
    # if not train_unet_only:
        # for p in models["gaussian1"].parameters():
        #     p.data.fill_(0.5)
        # for p in models["sigma1"].parameters():
        #     p.data.fill_(0.5)
        # for p in models["gaussian2"].parameters():
        #     p.data.fill_(0.5)
        # for p in models["sigma2"].parameters():
        #     p.data.fill_(0.5)
        
        # for g in range(1, gauss_number+1): 
            # parameters_to_train += list(models['sigma{}'.format(g)].parameters())
            # parameters_to_train += list(models['gaussian{}'.format(g)].parameters())
            
        # parameters_to_train += list(models["gaussian1"].parameters())
        # parameters_to_train += list(models["sigma1"].parameters())
        
        # parameters_to_train += list(models["gaussian2"].parameters())
        # parameters_to_train += list(models["sigma2"].parameters())

    # optimizer
    model_optimizer = optim.Adam(parameters_to_train, learning_rate)
    model_lr_scheduler = optim.lr_scheduler.StepLR(model_optimizer, scheduler_step_size, 0.1)

    if load_model:
        load_model_fxn('/code/code/3_combinedframe_recon_pretrained_trainable_dataaug_True_gauss_num_2_batchnorm_True_ssim_l1_0.55_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_19', ["decompose"], models)
        
   
 # dataloader 
    datasets_dict = {"endovis": datasets.LungRAWDataset}
    dataset = datasets_dict['endovis']

    fpath = os.path.join(os.path.dirname(__file__), "splits", "endovis", "{}_files_phantom.txt")

    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.png'

    num_train_samples = len(train_filenames)
    num_total_steps = num_train_samples // batch_size * num_epochs
    # data_augment = False
    train_dataset =  dataset(
        data_path, train_filenames,  height,  width,
        frame_ids, 4, is_train=True, img_ext=img_ext, adversarial_prior =  adversarial_prior, len_ct_depth_data = 2271, data_augment = data_aug)
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True, drop_last=True)
        
    val_dataset = dataset(data_path, val_filenames,  height,  width,frame_ids, 4, is_train=False, img_ext=img_ext, adversarial_prior = False, len_ct_depth_data = 0,  data_augment = data_aug) 
    val_loader = DataLoader(val_dataset,  batch_size, shuffle = True, drop_last=True)
    val_iter = iter( val_loader)

    models['decompose'].to(device)

    if not train_unet_only:
        for g in range(1, gauss_number+1): 
            models['sigma{}'.format(g)].to(device)
            models['gaussian{}'.format(g)].to(device) 
        # models['gaussian1'].to(device)
        # models['sigma1'].to(device)
        
        # models['gaussian2'].to(device)
        # models['sigma2'].to(device)

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
    save_frequency = 20
    custom_step = 0
    prev_error = 100000000
    prev_fid = 100000
    for  epoch in range(num_epochs):
        print("Training")
        
        custom_step+=1
        outputs = {}
        
        for batch_idx, inputs in enumerate(train_loader):
            
            models['decompose'].train()
            if not train_unet_only:
                for g in range(1, gauss_number+1): 
                    models['sigma{}'.format(g)].train()
                    models['gaussian{}'.format(g)].train() 
                # models['gaussian1'].train()
                # models['sigma1'].train()
                
                # models['gaussian2'].train()
                # models['sigma2'].train()
        
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)
                
            before_op_time = time.time()

         
            if discriminator:
                total_loss = {'reprojection_loss':0, 'l1': 0, 'ssim_loss':0, 'discriminator':0, 'discriminator_train':0}
                
            else:
                total_loss = {'reprojection_loss':0, 'l1': 0, 'ssim_loss':0}

            total_fid = {'fid':0}
            
            
            #fcn model output mixture proportions too 
            for frame_id in [0, -1, 1]:
                features                = models["decompose"](inputs["color_aug", frame_id, 0])
                outputs['decompose']    = features[1]
                
                sigma_out_combined        = models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                gaussian_mask1            = models["gaussian1"](sigma_out_combined[:, :4])
                gaussian_mask2            = models["gaussian1"](sigma_out_combined[:, 4:8])
                gaussian_mask3            = models["gaussian1"](sigma_out_combined[:, 8:12])
                gaussian_mask4            = models["gaussian1"](sigma_out_combined[:, 12:16])
                # gaussian_mask3            = models["gaussian1"](sigma_out_combined[:, :4])
                
                # gaussian_mask1            = models["gaussian1"](sigma_out_combined[:, :4])
                # gaussian_mask2            = models["gaussian2"](sigma_out_combined[:, 4:8])
                # gaussian_mask3            = models["gaussian3"](sigma_out_combined[:, 8:12])
                
                # outputs['compose'] = outputs['decompose'] * (sigma_out_combined[:,12][:,None, None, None]/3*gaussian_mask1[0] + sigma_out_combined[:,13][:,None, None, None]/3*gaussian_mask2[0] +sigma_out_combined[:,14][:,None, None, None]/3*gaussian_mask3[0])
                
                outputs['compose'] = outputs['decompose'] * (gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)

                if discriminator: 
                    total_loss['discriminator_train']+=compute_disc_loss(inputs["color_aug", frame_id, 0], outputs['compose'].detach())
                
                losses = compute_reprojection_loss(outputs['compose'], inputs["color_aug", frame_id, 0], frac)
                
                total_loss['l1']+=losses['l1']
                total_loss['reprojection_loss']+=losses['reprojection_loss']
                total_loss['ssim_loss']+=losses['ssim_loss']
                
                if discriminator:
                    total_loss['discriminator']=total_loss['discriminator']+losses['discriminator']
                    

                total_fid['fid']+=evaluation_FID(fid_criterion, inputs["color_aug", frame_id, 0], outputs['compose'])
                # total_fid['fid']+=computeFID(inputs["color", frame_id, 0], outputs['compose'], fid_criterion)
            
            # for 3 frames
            total_loss['l1']/=3
            total_loss['reprojection_loss']=total_loss['reprojection_loss']/3
            total_loss['ssim_loss']/=3
            total_fid['fid']/=3
            

            if discriminator:
                total_loss['discriminator']=total_loss['discriminator']/3
            
            model_optimizer.zero_grad()
            total_loss['reprojection_loss'].backward()
            # losses['reprojection_loss'].backward()
            model_optimizer.step()

            
            # discriminator train step
            if discriminator:
                total_loss['discriminator_train']/=3
                total_loss['discriminator_train'].backward()
                optimizer_Discriminator.step() 
                Discriminator.eval()
                
            duration = time.time() - before_op_time
            
            step+=1
            
            # save model
            # if losses['reprojection_loss'] < prev_error: 
            #     # save_model 
            #     save_model(epoch, 'code/{}'.format(experiment_name), models, model_optimizer)
            #     prev_error = losses['reprojection_loss']
            
            # save model
            if total_fid['fid'] < prev_fid: 
                # save_model 
                save_model(epoch, 'code/{}'.format(experiment_name), models, model_optimizer)
                prev_fid = total_fid['fid']
            
            # wand_b loggin 
            if ( step + 1) %  save_frequency == 0:
                with torch.no_grad():
                    models['decompose'].eval()
                    models['sigma_combined'].eval()
                    if not train_unet_only: 
                        # models['sigma1'].eval()
                        models['gaussian1'].eval()
                        
                        # models['sigma2'].eval()
                        # models['gaussian2'].eval()
                    
                    # features_val = models["decompose"](inputs["color_aug", 0, 0][0][None, :, :, :])
                    features_val        = models["decompose"](inputs["color_aug", 0, 0])
                    image               = features_val[1]
                    if not train_unet_only: 
                        sigma_out_val1                = models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                        gaussian_mask_val1            = models["gaussian1"](sigma_out_combined[:, :4])
                        gaussian_mask_val2            = models["gaussian1"](sigma_out_combined[:, 4:8])
                        
                        gaussian_mask_val3            = models["gaussian1"](sigma_out_combined[:, 8:12])
                        gaussian_mask_val4            = models["gaussian1"](sigma_out_combined[:, 12:16])

                        final_val = image * (gaussian_mask_val1[0]/4 + gaussian_mask_val2[0]/4 + gaussian_mask_val3[0]/4 + gaussian_mask_val4[0]/4)
                        
                        # final_val                     = image * (gaussian_mask_val1[0]/2 + gaussian_mask_val2[0])
                        # gaussian_mask_val3            = models["gaussian3"](sigma_out_combined[:, 8:12])
                        # final_val                     = image * (sigma_out_val1[:,12][:,None, None, None]/3*gaussian_mask_val1[0] + sigma_out_val1[:,13][:,None, None, None]/3*gaussian_mask_val2[0] +sigma_out_val1[:,14][:,None, None, None]/3*gaussian_mask_val3[0])
                        
                        # final_val           = image*gaussian_mask_val1[0]*gaussian_mask_val2[0]
                        
                        # sigma_out_val1      = models['sigma_combined'](features_val[0]) # check the dimension. pass through the convolutions. Input in the gaussian network 
                        # gaussian_mask_val1   = models["gaussian1"](sigma_out_val1)
                        
                        # sigma_out_val2       = models['sigma2'](features_val[0]) # check the dimension. pass through the convolutions. Input in the gaussian network 
                        # gaussian_mask_val2   = models["gaussian2"](sigma_out_val2)
                        # # final_val           = image*gaussian_mask_val[0].repeat(16,1,1,1)
                        # final_val           = image*gaussian_mask_val1[0]*gaussian_mask_val2[0]
                        
                        print('val')
                        # print(gaussian_mask_val[1])
                        print(sigma_out_val1[:4,:])
                        # print(sigma_out_val2[:4,:])
                    
                    wandb.log({"{}".format('train_original'):wandb.Image(inputs["color_aug", 0, 0], caption ='original image'),'custom_step':custom_step})  
                    wandb.log({"{}".format('train_intermediate'):wandb.Image(make_grid(image), caption = 'intermediate image'),'custom_step':custom_step})  
                    
                    if not train_unet_only: 
                        wandb.log({"{}".format('train_reconstructed'):wandb.Image(make_grid(final_val), caption = 'reconstructed image'),'custom_step':custom_step})  
                        wandb.log({"{}".format('train_gaussmask1'):wandb.Image(make_grid(gaussian_mask_val1[0]), caption = 'gaussian mask1'),'custom_step':custom_step})
                        wandb.log({"{}".format('train_gaussmask2'):wandb.Image(make_grid(gaussian_mask_val2[0]), caption = 'gaussian mask2'),'custom_step':custom_step}) 
                        wandb.log({"{}".format('train_gaussmask3'):wandb.Image(make_grid(gaussian_mask_val3[0]), caption = 'gaussian mask3'),'custom_step':custom_step}) 
                        wandb.log({"{}".format('train_gaussmask4'):wandb.Image(make_grid(gaussian_mask_val4[0]), caption = 'gaussian mask4'),'custom_step':custom_step})       
                        
                    
                    wandb.log({"{}".format('learning_rate'):model_lr_scheduler.optimizer.param_groups[0]['lr'],'custom_step':custom_step})
                    for l, v in total_loss.items():
                        wandb.log({"{}_{}".format('train', l):v, 'custom_step':custom_step})
                    
                    # FID 
                    for l, v in total_fid.items():
                        wandb.log({"{}_{}".format('train_fid', l):v, 'custom_step':custom_step})
                    
                        
                models['sigma_combined'].train()
                models['decompose'].train()
                models['gaussian1'].train()
                    
        model_lr_scheduler.step()
        #save model
        # save_model(epoch, 'code/logs', models, model_optimizer)
        
    wandb.finish()
        
        
