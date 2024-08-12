# Copyright Niantic 2019. Patent Pending. All rights reserved.import
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function


import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

# from datasets import SCAREDRAWDataset

# from datasets import LungRAWDataset

from datasets import endoSLAMRAWDataset

from torchvision.utils import save_image

import wandb_logging

import torchvision.transforms as transforms


from evaluate_pose import plotTrajectory, dump_xyz, dump_r, compute_ate, compute_re
# from torchviz import make_dot

class Trainer:
    def __init__(self, options, lr = 1e-6, sampling = 1, wandb_sweep = False, wandb_config = '', wandb_obj = None, frac = 0 ):
        
        self.opt = options
        print('learning rate {} sampling frequency : {}'.format(lr, sampling))
        
        gauss_static = self.gkern(192)
        gauss_static[gauss_static < 0.6] = 0
        gauss_static[gauss_static != 0.0] = 1.0
        self.gauss_static_mask = gauss_static
        # img_gauss  = gauss_static([0, 0, 1, 1])
            
            
        self.min_mean_trajectory_ates = 1000000
        self.min_mean_trajectory_res = 1000000
        self.min_std_trajectory_ates = 1000000
        self.min_std_trajectory_res = 1000000
        
        if options.wandb_sweep: 
            # self.wanb_obj = wandb_logging.wandb_logging(options)
            # self.wandb_config = self.wanb_obj.get_config()
            # self.sampling_frequency = self.wandb_config['sampling_frequency']
            # self.learning_rate      = self.wandb_config['learning_rate']
            
            self.wanb_obj = wandb_obj
            self.wandb_config = wandb_config
            self.sampling_frequency = self.wandb_config['sampling_frequency']
            self.learning_rate      = self.wandb_config['learning_rate']
            
        else:
            # self.wanb_obj = wandb_logging.wandb_logging(options)
            # self.sampling_frequency = self.opt.sampling_frequency
            # self.learning_rate = self.opt.learning_rate
            
            # parameter search 
            self.frac               = frac
            self.sampling_frequency = sampling
            self.learning_rate      = lr
            self.opt.frac           = frac
            self.opt.learning_rate  = lr
            self.opt.sampling_frequency = sampling
        self.wanb_obj               = wandb_logging.wandb_logging(self.opt, experiment_name = 'gaussTrain_{}_disc_prior_{}'.format(False, 'patchGAN'))

        self.writeFile(mode = "w")
        # set the manually the hyperparamters you want to optimize using sampling_frequency and learning rate
        
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name + '_{}'.format(self.learning_rate))

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.si_loss = SLlog()
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.pose_prior: 
            self.pose_criterion = nn.MSELoss()
        
        if self.opt.pose_consistency_loss:
            self.pose_consistency_criterion_rot = nn.MSELoss()
            self.pose_consistency_criterion_trans = nn.MSELoss()
        
        if self.opt.longterm_consistency_loss:
            self.longterm_consistency_criterion = nn.MSELoss()
        
        if self.opt.split == "endoSLAM":
            # self.depth_endoslam_loss = nn.MSELoss()
            self.depth_endoslam_loss = SLlog()
            
            
        if self.opt.adversarial_prior: 
            
            self.disc_response = {}
            # Define model 
            # input_shape = (1, self.opt.width, self.opt.height) # this we will have to check
            self.criterion_Discriminator = torch.nn.BCEWithLogitsLoss()
            
            if self.opt.multiscale_adversarial_prior: 
                
                for i in range(len(self.opt.scales)):
                    input_shape[i] = (1, self.opt.width//2, self.opt.height//2)
                    self.Discriminator[i] = networks.Discriminator(input_shape[i])
                    # self.Discriminator[i] = networks.DiscriminatorUnet(input_shape[i])
                    self.Discriminator[i].to(self.device)
                    
                     # Adversarial ground truths
                    #  valid = Variable(Tensor(np.ones((real_A.size(0), *D_A2.output_shape))), requires_grad=False)
                    self.valid[i] = torch.ones((1, *self.Discriminator.output_shape), requires_grad=False).repeat([self.opt.batch_size, 1, 1, 1]).to(self.device)
                    self.fake[i] = torch.zeros((1, *self.Discriminator.output_shape), requires_grad=False).repeat([self.opt.batch_size, 1, 1, 1]).to(self.device)
                    
                    self.optimizer_Discriminator[i] = torch.optim.Adam(self.Discriminator[i].parameters(), lr=self.opt.discriminator_lr, betas=(self.opt.b1, self.opt.b2))
            else:
                input_shape = (1, self.opt.width, self.opt.height) # this we will have to check
                # self.Discriminator = networks.DiscriminatorUnet(input_shape)
                self.Discriminator = networks.Discriminator(input_shape)
                self.Discriminator.to(self.device)
                
                # Adversarial ground truths
                #  valid = Variable(Tensor(np.ones((real_A.size(0), *D_A2.output_shape))), requires_grad=False)
                self.valid = torch.ones((1, *self.Discriminator.output_shape), requires_grad=False).repeat([self.opt.batch_size, 1, 1, 1]).to(self.device)
                self.fake = torch.zeros((1, *self.Discriminator.output_shape), requires_grad=False).repeat([self.opt.batch_size, 1, 1, 1]).to(self.device)
                
                self.optimizer_Discriminator = torch.optim.Adam(self.Discriminator.parameters(), lr=self.opt.discriminator_lr, betas=(self.opt.b1, self.opt.b2))
        
        if self.opt.optical_flow:
            self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
            self.models["position_encoder"].to(self.device)

            self.models["position"] = networks.PositionDecoder(
                self.models["position_encoder"].num_ch_enc, self.opt.scales)
            self.models["position"].to(self.device)
            
            self.parameters_to_train += list(self.models["position_encoder"].parameters())
            self.parameters_to_train += list(self.models["position"].parameters())
            
            self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
            self.spatial_transform.to(self.device)

            self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
            self.get_occu_mask_backward.to(self.device)

            self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
            self.get_occu_mask_bidirection.to(self.device)
            
            for scale in self.opt.scales:
                self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
                self.position_depth[scale].to(self.device)
                
        
        self.resize_transform = {}
        for s in self.opt.scales:
            self.resize_transform[s] = Resize((192// 2 ** s,192// 2 ** s))
            
        if self.opt.enable_gauss_mask:
            # self.gauss_parameters_to_train = []
            self.models['decompose'] = networks.UNet(3, 3)
            
            self.models['sigma_combined'] = networks.FCN(output_size = 16) # 4 sigma and mu x 3 for 3 gaussians
            self.models['gaussian{}'.format(1)] = networks.GaussianLayer(192)
          
            self.models['decompose'].to(self.device)
            self.models['sigma_combined'].to(self.device)
            self.models['gaussian{}'.format(1)].to(self.device)
            
            # train gaussian 
            self.set_eval_gauss()
            
        if self.opt.gaussian_correction:
            
            self.resize_transform = {}
            for s in self.opt.scales:
                self.resize_transform[s] = Resize((192// 2 ** s,192// 2 ** s))
                
            # self.gauss_parameters_to_train = []
            self.models['decompose'] = networks.UNet(3, 3)
            
            self.models['sigma_combined'] = networks.FCN(output_size = 16) # 4 sigma and mu x 3 for 3 gaussians
            self.models['gaussian{}'.format(1)] = networks.GaussianLayer(192)
    
            self.models['decompose'].to(self.device)
            self.models['sigma_combined'].to(self.device)
            self.models['gaussian{}'.format(1)].to(self.device)

            # self.parameters_to_train += list(self.models["decompose"].parameters())
            # self.parameters_to_train += list(self.models["sigma_combined"].parameters())
            # self.parameters_to_train += list(self.models["gaussian1"].parameters())
            
           
            self.set_eval_gauss()
            # train gaussian 
            # self.set_train_gauss()

    
            # self.criterion_Discriminator = FrechetInceptionDistance(feature=64)

            # self.disc_model_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer_Discriminator, self.opt.scheduler_step_size, 0.1)
       
            # loss_real = criterion_GAN(Discriminator(real_A), valid)
            # # Fake loss (on batch of previously generated samples)
            # fake_A2_ = fake_A2_buffer.push_and_pop(fake_A2)
            # loss_fake = criterion_GAN(D_A2(fake_A2_.detach()), fake)

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.pre_trained_generator:
            n_residual_blocks = 9 
            channels = 1
            network_name = "3cGAN"
            dataset_name = "ex-vivo"
            epoch_name = 50
            input_shape = (channels, self.opt.width, self.opt.height) # this we will have to check
            
            self.models["pre_trained_generator"] = networks.GeneratorResNet(input_shape, n_residual_blocks)
            self.models["pre_trained_generator"].to(self.device)
            
            self.models["pre_trained_generator"].load_state_dict(torch.load("saved_models/%s-%s-G_AB-%dep.pth" % (network_name, dataset_name, epoch_name)))
            
            self.gen_transform = transforms.Grayscale()
            
            self.depth_gan_log_loss = SLlog()
            # self.normalize_transform = transforms.Normalize(0.5, 0.5)
            
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        # self.model_lr_scheduler = optim.lr_scheduler.StepLR(
        #     self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        # datasets_dict = {"kitti": datasets.KITTIRAWDataset,
        #                  "kitti_odom": datasets.KITTIOdomDataset}

        if self.opt.split == "endoSLAM":  
            datasets_dict = {"endovis": datasets.endoSLAMRAWDataset}
        else:
            datasets_dict = {"endovis": datasets.LungRAWDataset}
            


        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.pose_prior:
            fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files_phantom_sampling_freq_5_pose_explicit.txt")
        else:
            if self.opt.split == "endoSLAM": 
                fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_ColonendoSLAMUnity.txt")
            else:
                # fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files_phantom_sampling_freq_5.txt")
                fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "phantom_all_{}_fw.txt")
        
        train_filenames = readlines(fpath.format("train")) # exclude frame accordingly # change this 
        val_filenames = readlines(fpath.format("val"))

        # train_filenames = readlines(fpath.format("train"))[self.sampling_frequency+2:-self.sampling_frequency-6] # exclude frame accordingly
        # val_filenames = readlines(fpath.format("val"))[self.sampling_frequency+2:-self.sampling_frequency-6]
              
        img_ext = '.png'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, data_augment = True, is_train=True, img_ext=img_ext, adversarial_prior = self.opt.adversarial_prior, len_ct_depth_data = 2082, sampling_frequency = self.sampling_frequency, pose_prior = self.opt.pose_prior )
        
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True, drop_last=True)
        
        # self.train_loader = DataLoader(
        #     train_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        # NOTE: sampling_frequency = 2 shoudl be two to obtain a frequency of 1
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, data_augment = False, img_ext=img_ext, adversarial_prior = False, len_ct_depth_data = 0, sampling_frequency = 2, pose_prior = self.opt.pose_prior)
        
        # self.val_loader = DataLoader(
        #     val_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        # for pose trajectory
        if self.opt.eval_pose_trajectory:  
            fpath_trajectory = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "test_files_phantom_{}.txt")
            fpath_gt = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "gt_poses_phantom_{}.npz")
            
            # for trajectory plot add trajectory 1 and 14 
            val_traj_14_filenames = readlines(fpath_trajectory.format("14"))[0:50] # not used in training 
            val_traj_1_filenames  = readlines(fpath_trajectory.format("1"))[0:50] # used in training
            
            
            val_trajectory_14_dataset = self.dataset(
            self.opt.data_path, val_traj_14_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, adversarial_prior = False, len_ct_depth_data = 0)
        
            val_trajectory_1_dataset = self.dataset(
                self.opt.data_path, val_traj_1_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, adversarial_prior = False, len_ct_depth_data = 0)
            
            self.val_traj_14_loader = DataLoader(
                val_trajectory_14_dataset, self.opt.batch_size, shuffle=False, drop_last=True)
            
            self.val_traj_1_loader = DataLoader(
                val_trajectory_1_dataset, self.opt.batch_size, shuffle=False, drop_last=True)
            
            # self.val_iter_1_traj = iter(self.val_traj_1_loader)
            # self.val_iter_14_traj = iter(self.val_traj_14_loader)

            
            # gt poses 
            self.gt_local_poses_14 = np.load(fpath_gt.format("13"), fix_imports=True, encoding='latin1')["data"]
            self.gt_local_poses_1 = np.load(fpath_gt.format("1"), fix_imports=True, encoding='latin1')["data"]
            
        
        # self.writers = {}
        # for mode in ["train", "val"]:
        #     self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        
    def gaussian_fn(self, M, std, mean):
        #mean 0 - 1
    
        # n = torch.arange(0, M) - (M - 1.0) / 2.0
        # n = torch.arange(0, M).to('cuda') - mean
        # sig2 = 2 * std * std
        # n = n.to('cuda')
        # sig2.to('cuda')
        # w = torch.exp(-n ** 2 / (sig2 + 1e-7)).to('cuda')
        
        
        n = torch.arange(0, M) - mean
        sig2 = 2 * std * std
        w = torch.exp(-n ** 2 / (sig2 + 1e-7))
        
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
        gkern2d = gkern2d.expand(3, kernlen, kernlen)
        return gkern2d
    
    def set_train_gauss(self):
        self.models['decompose'].train()
        self.models['sigma_combined'].train()
        self.models['gaussian{}'.format(1)].train()
        # self.models['decompose'].train()
        # self.models['sigma1'].train()
        # self.models['gaussian1'].train()
        # if self.opts.gauss_number > 1:
        #     self.models['sigma2'].train()
        #     self.models['gaussian2'].train()
        
    def set_eval_gauss(self):
        self.models['decompose'].eval()
        self.models['sigma_combined'].eval()
        self.models['gaussian{}'.format(1)].eval()
            
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def writeFile(self, mode = "a"):
        f = open("trajectory_errors_{}_{}.txt".format(self.opt.model_name, self.learning_rate), mode)
        
        f.write("mean_ates {} \n".format(self.min_mean_trajectory_ates))
        f.write("mean_res {} \n".format(self.min_mean_trajectory_res))
        f.write("std_ates {} \n".format(self.min_std_trajectory_ates))
        f.write("std_res {} \n".format(self.min_std_trajectory_res))
        f.close()
        return 
    
    def train(self):
        """Run the entire training pipeline
        """
        # with torch.autograd.set_detect_anomaly(True): 
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            
            if self.opt.eval_pose_trajectory:
                traj_outputs, traj_losses = self.get_trajectory_error(self.val_traj_14_loader, self.gt_local_poses_14)
                
                if traj_losses['mean_ates'] < self.min_mean_trajectory_res:
                    self.min_mean_trajectory_ates = traj_losses['mean_ates']
                    self.min_mean_trajectory_res = traj_losses['mean_res']
                    self.min_std_trajectory_ates = traj_losses['std_ates']
                    
                    self.writeFile()
                    
                self.log_wand("val2", traj_outputs, traj_losses, self.wanb_obj, step = self.epoch, character="trajectory")
            
            if self.epoch >= 0 and (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
        self.wanb_obj.finishWandb()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        gaussian_reponse = None

        d_loss = 0
        num_run = 0 
        for batch_idx, inputs in enumerate(self.train_loader):

            num_run+=1
            before_op_time = time.time()
            
            # pass all inputs through the 
            # assumption that frame to gauss mask is the same
            # pass the input1 to the tained unet -> 1- decomposed 2- get sigma -> gaussian 
            # pass this decomposed to depth network. 
            # 
            
            if self.opt.enable_gauss_mask:
                gaussian_reponse = {'gaussian_mask1':[], 'original':[]}
                self.set_eval_gauss()
                # use a precomputer network. 
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                    
                # combine gauss masks of all three frames
                gauss_mask_combined = []
                for frame_id in self.opt.frame_ids:
                    features      = self.models["decompose"](inputs["color_aug", frame_id, 0]) # no augmentation for validation 
                    decomposed    = features[1]
                    
                    sigma_out_combined        = self.models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                    gaussian_mask1            = self.models["gaussian1"](sigma_out_combined[:, :4])
                    gaussian_mask2            = self.models["gaussian1"](sigma_out_combined[:, 4:8])
                    gaussian_mask3            = self.models["gaussian1"](sigma_out_combined[:, 8:12])
                    gaussian_mask4            = self.models["gaussian1"](sigma_out_combined[:, 12:16])
                    
                    gauss_mask_combined.append(gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)
                
                mask, idx = torch.min(torch.cat(gauss_mask_combined, 1), 1, keepdim = True) 
                # mask = torch.cat(gauss_mask_combined, 1).sum(1)/9
                
                # mask[mask < 0.6] = 0
                mask[mask < 0.7] = 0
                
                gaussian_reponse['gaussian_mask1'].append(mask[:,None, :, :][:4]) 
                # gaussian_reponse['decomposed'].append(inputs["color_aug", frame_id, 0])
                gaussian_reponse['original'].append(inputs["color_aug", frame_id, 0])
                    
                for s in self.opt.scales:
                    # inputs["color_aug", frame_id, s] = self.resize_transform[s](decomposed)
                    inputs.update({("gauss_mask", s):self.resize_transform[s](mask)})
                    mask_t = torch.ones(inputs[("gauss_mask", s)].shape).cuda()
                    new_mask = inputs[("gauss_mask", s)]
                    mask_t[new_mask == 0] = 0
                    # mask_t[new_mask != 0] = 1
                    
                    # multiply inputs with it: 
                    for frame_id in self.opt.frame_ids:
                        inputs["color_aug_original", frame_id, s] = inputs["color_aug", frame_id, s]
                        inputs["color_aug", frame_id, s]=inputs["color_aug", frame_id, s]*mask_t # wrong there shouudl be no gradation
                        # save_image(inputs["color_aug", frame_id, s], f'image_{s}_fid_{frame_id}.png')
            
            if self.opt.enable_gauss_static_mask:
                for s in self.opt.scales:
                    inputs.update({("gauss_mask", s):self.resize_transform[s](self.gauss_static_mask)})
                    # mask_t = torch.ones(inputs[("gauss_mask", s)].shape).cuda()
                    new_mask = inputs[("gauss_mask", s)]
                    # mask_t[new_mask == 0] = 0
                    # mask_t[new_mask != 0] = 1
                    
                    # multiply inputs with it: 
                    for frame_id in self.opt.frame_ids:
                        inputs["color_aug_original", frame_id, s] = inputs["color_aug", frame_id, s]
                        inputs["color_aug", frame_id, s]=inputs["color_aug", frame_id, s]*new_mask # wrong there shouudl be no gradation
                
                
            if self.opt.gaussian_correction: 
                # self.set_train_gauss()
                self.set_eval_gauss()
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                
                gauss_mask_combined = []
                gaussian_reponse = {'gaussian_mask1':[], 'gaussian_mask2':[], 'gaussian_mask3':[], 'gaussian_mask4':[], 'reconstructed':[], 'decomposed':[], 'original':[]}
                for frame_id in self.opt.frame_ids:
                    features      = self.models["decompose"](inputs["color_aug", frame_id, 0]) # no augmentation for validation 
                    decomposed    = features[1]
                    
                    sigma_out_combined        = self.models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                    gaussian_mask1            = self.models["gaussian1"](sigma_out_combined[:, :4])
                    gaussian_mask2            = self.models["gaussian1"](sigma_out_combined[:, 4:8])
                    gaussian_mask3            = self.models["gaussian1"](sigma_out_combined[:, 8:12])
                    gaussian_mask4            = self.models["gaussian1"](sigma_out_combined[:, 12:16])
                    
                    gauss_mask_combined.append(gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)
                    # re_composed = decomposed * (gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)

                    # gaussian_reponse['original'].append(inputs["color_aug", frame_id, 0][0, :, :, :]) 
                    # update the images
                    inputs.update({("gaussian_mask1", frame_id, 0):gaussian_mask1[0]})
                    inputs.update({("gaussian_mask2", frame_id, 0):gaussian_mask2[0]})
                    inputs.update({("gaussian_mask3", frame_id, 0):gaussian_mask3[0]})
                    inputs.update({("gaussian_mask4", frame_id, 0):gaussian_mask4[0]})
                    
                    for s in self.opt.scales:
                        # inputs["color_aug", frame_id, s] = self.resize_transform[s](decomposed)
                        inputs.update({("color_aug_decompose", frame_id, s):self.resize_transform[s](decomposed)})
                        # inputs.update({("gaussian_mask1", frame_id, s):self.resize_transform[s](gaussian_mask1[0])})
                        # inputs.update({("gaussian_mask2", frame_id, s):self.resize_transform[s](gaussian_mask2[0])})
                   
                
                    # if self.opts.gauss_number > 1:
                    #     inputs.update({("gaussian_mask2", frame_id, 0):gaussian_mask2[0]})
                    
                    # multiply gaussian mask with at the periphery 
                    
                
                mask = torch.cat(gauss_mask_combined, 1).sum(1)/9
                mask[mask < 0.6] = 0
                # mask[mask >= 0.5] = 1
                inputs.update({("gaussian_remove_mask", 0):mask[:,None, :, :]})

                gaussian_reponse['gaussian_mask1'].append(gaussian_mask1[0][0, :, :, :]) 
                gaussian_reponse['gaussian_mask2'].append(gaussian_mask2[0][0, :, :, :])
                gaussian_reponse['gaussian_mask3'].append(gaussian_mask3[0][0, :, :, :])
                gaussian_reponse['gaussian_mask4'].append(gaussian_mask4[0][0, :, :, :])
                
                # if self.opts.gauss_number > 1:
                #     gaussian_reponse['gaussian_mask2'].append(gaussian_mask2[0][0, :, :, :]) 
                # gaussian_reponse['reconstructed'].append(re_composed[0, :, :, :]) 
                gaussian_reponse['decomposed'].append(decomposed[0, :, :, :]) 
                    
            # caculate the loss using decomposed output 
            if self.opt.gaussian_correction:
                outputs, losses = self.process_batch_gauss(inputs)
            else:
                if self.opt.enable_endoMasking:
                    for s in self.opt.scales:

                    # multiply inputs with it: 
                        for frame_id in self.opt.frame_ids:
                            mask_endo = torch.ones(inputs["color", frame_id, s].shape)
                            idx, mask_ = inputs["color", frame_id, s].min(1)
                            mask_ = mask_[:, None, :, :]
                            mask_new = torch.cat([mask_, mask_, mask_], 1)
                            mask_endo[mask_new==0] = 0 
                            inputs["color_aug", frame_id, s] = inputs["color_aug", frame_id, s]*mask_endo
                            
                            # save_image(inputs["color_aug", frame_id, s], "inputs[color_aug, {}, {}].png".format(frame_id, s))

                            inputs.update({("endoslam_mask", s):mask_endo})
                            # save_image(inputs["endoslam_mask", s], "inputs[endoslam_mask, {}].png".format(s), normalize = True)
                              
                outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            
            # backward step with the gaussian networks
            if self.opt.adversarial_prior:
                d_loss+=self.discriminator_train_step(inputs,outputs)
            
                # loss_real = criterion_GAN(Discriminator(real_A), valid)
            # # Fake loss (on batch of previously generated samples)
            # fake_A2_ = fake_A2_buffer.push_and_pop(fake_A2)
            # loss_fake = criterion_GAN(D_A2(fake_A2_.detach()), fake)
                
                
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            # late_phase = self.step % 2000 == 0
            phase = batch_idx % self.opt.log_frequency == 0
            
            if phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.set_eval()
                if self.opt.gaussian_correction:
                    self.set_eval_gauss()
                with torch.no_grad():
                    # self.log_wand("train2", outputs, losses, self.wanb_obj, step = self.epoch, character="disp", lr = self.model_lr_scheduler.optimizer.param_groups[0]['lr'])
                    
                    self.log_wand("train2", outputs, losses, self.wanb_obj, step = self.epoch, character="disp", lr = 1., 
                                    use_discriminator_loss=False, discriminator_loss= 0, discriminator_response=None, 
                                    gaussian_decomposition=self.opt.gaussian_correction, gaussian_response=gaussian_reponse)
                    
                    if self.opt.adversarial_prior:
                        
                        if self.opt.gaussian_correction:
                            self.log_wand("train2", outputs, losses, self.wanb_obj, step = self.epoch, character="disp", lr = 1., 
                                      use_discriminator_loss=True, discriminator_loss=d_loss/num_run, discriminator_response=self.disc_response, 
                                      gaussian_decomposition=self.opt.gaussian_correction, gaussian_response=gaussian_reponse)
                            
                        else:
                            self.log_wand("train2", outputs, losses, self.wanb_obj, step = self.epoch, character="disp", lr = 1., 
                                      use_discriminator_loss=True, discriminator_loss=d_loss/num_run, discriminator_response=self.disc_response)
                    else:
                        self.log_wand("train2", outputs, losses, self.wanb_obj, step = self.epoch, character="disp", lr = 1.)
                    # print('learning_rate_{}'.format(self.model_lr_scheduler.optimizer.param_groups[0]['lr']))
                self.val()
                self.set_train()
                # self.log("train", inputs, outputs, losses)
                
                # self.set_train_gauss()
                if self.opt.gaussian_correction:
                    self.set_eval_gauss()
                

            self.step += 1
        
        print('disc_loss_{}'.format(d_loss/num_run))
        # self.model_lr_scheduler.step()

    def discriminator_train_step(self, inputs, outputs):
        
        # for key, ipt in inputs.items():
        #     inputs[key] = ipt.to(self.device)
            
        # backpropagate through discriminator
        self.Discriminator.train()
        self.optimizer_Discriminator.zero_grad()

        ct_loss_disc = self.Discriminator(inputs[('ct_prior', 0)]*inputs[("gauss_mask", 0)].detach()) # check size for valid. inputs[("gauss_mask", s)]
        loss_real = self.criterion_Discriminator(ct_loss_disc, self.valid)
        
        self.disc_response[('disc_response_ct')] = ct_loss_disc
        
        loss_fake = 0 
        for scale in self.opt.scales:
            depth_disc_res = self.Discriminator((outputs[("depth", 0, scale)]*inputs[("gauss_mask", 0)]).detach())
            self.disc_response[('disc_response', scale)] = depth_disc_res
            loss_fake+=self.criterion_Discriminator(depth_disc_res, self.fake)
            
        loss_fake=loss_fake/len(self.opt.scales)
        
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        
        self.optimizer_Discriminator.step()
        
        self.Discriminator.eval()
        
        # train all discriminators together 
        if self.opt.multiscale_adversarial_prior: 
            
            for i in range(len(self.opt.scales)):
                self.Discriminator[i].train()
                self.optimizer_Discriminator[i].zero_grad()

                ct_loss_disc = self.Discriminator[i](inputs[('ct_prior', 0)]//2**i)
                loss_real[i] = self.criterion_Discriminator(ct_loss_disc, self.valid[i])
                
                # this change we want scaled depths
                loss_fake[i] = self.criterion_Discriminator(self.Discriminator[i](outputs[("depth", 0, i)].detach()), self.fake[i])
                # Total loss
                loss_D[i] = (loss_real[i] + loss_fake[i]) / 2

                loss_D[i].backward()
                self.optimizer_Discriminator[i].step()
                
                self.Discriminator[i].eval()
            
        
        # gan loss: how far the generation is from the real
        # (depth map - 1.0 ) + 
        
        return loss_D
            
            
    def get_trajectory_error(self, dataloader, gt_poses):
        
        self.set_eval()
        # get prediction of trajectory 
        outputs = {'trajectory':0}
        losses = {'mean_ates':0, 'std_ates':0, 'mean_res':0,'std_res':0 }
        
       
        # if self.opt.eval_pose_trajectory:
        pred_poses = []
        with torch.no_grad():
            for inputs in dataloader:
                
                
                # count = count + 1
                # print(count)
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()

                if self.opt.enable_gauss_mask:
                    
                    self.set_eval_gauss()
                    
                    gauss_mask_combined = []
                    for frame_id in [1,0]:
                        features      = self.models["decompose"](inputs["color", frame_id, 0]) # no augmentation for validation 
                        decomposed    = features[1]
                        
                        sigma_out_combined        = self.models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                        gaussian_mask1            = self.models["gaussian1"](sigma_out_combined[:, :4])
                        gaussian_mask2            = self.models["gaussian1"](sigma_out_combined[:, 4:8])
                        gaussian_mask3            = self.models["gaussian1"](sigma_out_combined[:, 8:12])
                        gaussian_mask4            = self.models["gaussian1"](sigma_out_combined[:, 12:16])
                        
                        gauss_mask_combined.append(gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)
                    
                    mask = torch.cat(gauss_mask_combined, 1).sum(1)/6
                    mask[mask<0.6] = 0
                    
                    mask_apply = (mask[:,None, :, :])
                    for frame_id in [0,1]:
                        
                        mask_t = torch.ones(mask_apply.shape).cuda()
                        new_mask = mask_apply
                        mask_t[new_mask == 0] = 0
                        inputs["color", frame_id, 0]=inputs["color", frame_id, 0]*mask_t
                # transforms.ToPILImage()(inputs[("color", 1, 0)].cpu().squeeze()).save('trainer_2_1.png')
                
                # modify according to gauss 
                all_color_aug = torch.cat([inputs[("color", 0, 0)], inputs[("color", 1, 0)]], 1)

                # save image here all color_aug channel 0 and channel 1
                features = [self.models["pose_encoder"](all_color_aug)]
                axisangle, translation = self.models["pose"](features)

                if self.opt.use_euler:
                    pred_poses.append(
                        transformation_from_parameters_euler(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                else:
                    
                    pred_poses.append(
                        transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                
        # predict errors            
        pred_poses = np.concatenate(pred_poses)
        
        outputs['trajectory'] = plotTrajectory(pred_poses, gt_poses)
        
        ates = []
        res = []
        # num_frames = gt_local_poses.shape[0]
        num_frames = pred_poses.shape[0] - 3
        track_length = 5
        
        for i in range(0, num_frames - 1):
            local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
            gt_local_xyzs = np.array(dump_xyz(gt_poses[i:i + track_length - 1]))
            local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
            gt_rs = np.array(dump_r(gt_poses[i:i + track_length - 1]))
            # if i + track_length - 1 > 50:
            #     print('here')
            # print(i + track_length - 1)

            ates.append(compute_ate(gt_local_xyzs, local_xyzs))
            res.append(compute_re(local_rs, gt_rs))

        # log this to wandb based on the 
        mean_ates, std_ates = np.mean(ates), np.std(ates)
        mean_res, std_res   = np.mean(res), np.std(res)
        losses['mean_ates'] = mean_ates
        losses['std_ates']  = std_ates
        losses['mean_res']  = mean_res
        losses['std_res']   = std_res
        
        self.set_train()
        
        return outputs, losses


    def process_batch_gauss(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        # we want to go from 1 -> 0 and -1 to 0 
        # the 
        features = self.models["encoder"](inputs["color_aug_decompose", 0, 0])
        outputs = self.models["depth"](features)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug_decompose", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug_decompose", 0, 0])
            outputs = self.models["depth"](features)

   
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # predict pose using gauss
        if self.use_pose_net:
            outputs.update(self.predict_poses_gauss(inputs, features))

        self.generate_images_pred_gauss(inputs, outputs)
        
        losses = self.compute_losses_gauss(inputs, outputs)

        return outputs, losses 
    
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        # we want to go from 1 -> 0 and -1 to 0 
        # the 
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

   
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses_gauss(self, inputs, features):
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug_decompose", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    if self.opt.use_euler:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_euler(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    else:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug_decompose", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    if self.opt.use_euler:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_euler(
                            axisangle[:, i], translation[:, i])
                    else:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, i], translation[:, i])

        return outputs
    
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]] # [-1, 0]
                        
                        # calculating reverse too
                        inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]
                        
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]] # [0, 1]
                        
                        # calculating reverse too
                        inputs_all_reverse = [pose_feats[f_i], pose_feats[0]]
                        
                    if self.opt.optical_flow:
                        position_inputs = self.models["position_encoder"](torch.cat(pose_inputs, 1))
                        position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                        outputs_0 = self.models["position"](position_inputs)
                        outputs_1 = self.models["position"](position_inputs_reverse)
                        
                        for scale in self.opt.scales:
                            # optical flow at every scale 
                            # check forward backward occlussion masks 
                            outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                            outputs[("position", "high", scale, f_i)] = F.interpolate(
                                outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                            
                            # change spatial transformer # shoudl this be equalivalent to the next frame ?
                            outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)], outputs[("position", "high", scale, f_i)])

                            outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                            outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                                outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                            
                            outputs[("occu_mask_backward", scale, f_i)],  outputs[("occu_map_backward", scale, f_i)]= self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                            outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(outputs[("position", "high", scale, f_i)],
                                                                                                            outputs[("position_reverse", "high", scale, f_i)])
                            
                   
                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    if self.opt.pose_consistency_loss:
                        reverse_pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all_reverse, 1))]
                        axisangle_reverse, translation_reverse = self.models["pose"](reverse_pose_inputs)
                        outputs[("reverse_axisangle", 0, f_i)] = axisangle_reverse
                        outputs[("reverse_translation", 0, f_i)] = translation_reverse
                    

                    # Invert the matrix if the frame id is negative
                    if self.opt.use_euler:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_euler(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    else:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
            
            if self.opt.longterm_consistency_loss:
                pose_inputs_longterm = [pose_feats[-1], pose_feats[1]]
                longterm_pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs_longterm, 1))]
                axisangle_longterm, translation_longterm = self.models["pose"](longterm_pose_inputs)
                outputs[("longterm_axisangle")] = axisangle_longterm
                outputs[("longterm_translation")] = translation_longterm
                
                outputs[("cam_T_cam_longterm")] = transformation_from_parameters_euler(axisangle_longterm[:, 0], translation_longterm[:, 0])
                outputs[("cam_T_cam_serial")] = torch.matmul(transformation_from_parameters_euler(outputs[("axisangle", 0, -1)][:, 0], outputs[("translation", 0, -1)][:, 0]), 
                                                             transformation_from_parameters_euler(outputs[("axisangle", 0, 1)][:, 0], outputs[("translation", 0, 1)][:, 0]))
                
                # convert to euler and minimize
                # outputs[("eulerTanslation_lonterm")] = matrix_to_euler_angles(outputs[("cam_T_cam_longterm")], 'ZYX')
                # outputs[("eulerTanslation_serial")] = matrix_to_euler_angles(outputs[("cam_T_cam_serial")], 'ZYX')
                
                # check to convert to rotation and translation 
                # outputs[("eulerTanslation_lonterm")] = torch.flatten(outputs[("cam_T_cam_longterm")], 1)
                # outputs[("eulerTanslation_serial")]  = torch.flatten(outputs[("cam_T_cam_serial")], 1)
                
                outputs[("eulerTanslation_lonterm")] = matrix_2_euler_vector(outputs[("cam_T_cam_longterm")], 'ZYX')
                outputs[("eulerTanslation_serial")]  = matrix_2_euler_vector(outputs[("cam_T_cam_serial")], 'ZYX')
                                       
        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    
                    if self.opt.use_euler:
                            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_euler(
                            axisangle[:, i], translation[:, i])
                    else:
                        
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:

            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            # inputs = self.val_iter.next()
            inputs = next(self.val_iter)

        with torch.no_grad():
            
            if self.opt.enable_gauss_static_mask:
                
                # use a precomputer network. 
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                    
                for s in self.opt.scales:
                    inputs.update({("gauss_mask", s):self.resize_transform[s](self.gauss_static_mask)})
                    # mask_t = torch.ones(inputs[("gauss_mask", s)].shape).cuda()
                    new_mask = inputs[("gauss_mask", s)].to(self.device)
                    # mask_t[new_mask == 0] = 0
                    # mask_t[new_mask != 0] = 1
                    
                    # multiply inputs with it: 
                    for frame_id in self.opt.frame_ids:
                        inputs["color_aug_original", frame_id, s] = inputs["color_aug", frame_id, s]
                        inputs["color_aug", frame_id, s]=inputs["color_aug", frame_id, s]*new_mask # wrong there shouudl be no gradation
                        
            if self.opt.enable_gauss_mask:
                gaussian_reponse = None
                self.set_eval_gauss()
                # use a precomputer network. 
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                    
                # combine gauss masks of all three frames
                gauss_mask_combined = []
                for frame_id in self.opt.frame_ids:
                    features      = self.models["decompose"](inputs["color_aug", frame_id, 0]) # no augmentation for validation 
                    decomposed    = features[1]
                    
                    sigma_out_combined        = self.models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                    gaussian_mask1            = self.models["gaussian1"](sigma_out_combined[:, :4])
                    gaussian_mask2            = self.models["gaussian1"](sigma_out_combined[:, 4:8])
                    gaussian_mask3            = self.models["gaussian1"](sigma_out_combined[:, 8:12])
                    gaussian_mask4            = self.models["gaussian1"](sigma_out_combined[:, 12:16])
                    
                    gauss_mask_combined.append(gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)
                
                mask = torch.cat(gauss_mask_combined, 1).sum(1)/9
                mask[mask < 0.7] = 0
                for s in self.opt.scales:
                    # inputs["color_aug", frame_id, s] = self.resize_transform[s](decomposed)
                    inputs.update({("gauss_mask", s):self.resize_transform[s](mask[:,None, :, :])})
                    mask_t = torch.ones(inputs[("gauss_mask", s)].shape).cuda()
                    new_mask = inputs[("gauss_mask", s)]
                    mask_t[new_mask == 0] = 0
                    for frame_id in self.opt.frame_ids:
                        inputs["color_aug_original", frame_id, s] = inputs["color_aug", frame_id, s]
                        inputs["color_aug", frame_id, s]=inputs["color_aug", frame_id, s]*mask_t
                        
            if self.opt.enable_endoMasking:
                for s in self.opt.scales:

                # multiply inputs with it: 
                    for frame_id in self.opt.frame_ids:
          
                        mask_endo = torch.ones(inputs["color", frame_id, s].shape)
                        idx, mask_ = inputs["color", frame_id, s].min(1)
                        mask_ = mask_[:, None, :, :]
                        mask_new = torch.cat([mask_, mask_, mask_], 1)
                        mask_endo[mask_new==0] = 0
                        
                        inputs["color_aug", frame_id, s] = inputs["color_aug", frame_id, s]*mask_endo
                        
                        inputs.update({("endoslam_mask", s):mask_endo})
            outputs, losses = self.process_batch(inputs) # process batch eval 
            
            # run on trajectory for pose
            # get pose prediction. 
            # calculate errors on 2 trajectories. 
            # calculate error on GT yaw and pitch relative 
            # plot trajectories wrt to GT EM tracker data
           
            if self.opt.eval_pose_trajectory:
                traj_outputs_train, traj_losses_train = self.get_trajectory_error(self.val_traj_1_loader, self.gt_local_poses_1)
                self.log_wand("train_val", traj_outputs_train, traj_losses_train, self.wanb_obj, step = self.epoch, character="trajectory")

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            # self.log_wand("val2", outputs, losses, self.wanb_obj, step = self.epoch, character="disp", lr = self.model_lr_scheduler.get_last_lr()[0])
            self.log_wand("val2", outputs, losses, self.wanb_obj, step = self.epoch, character="disp", lr = 1.)
            
            # self.log("val", inputs, outputs, losses)
            
            # wand 
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred_gauss(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) # this should be 0-100
            
            # depth = disp_to_depth_no_scaling(disp) # this should be 0-1

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    if self.opt.use_euler:
                         
                        
                        T = transformation_from_parameters_euler(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0) # why it is multiplied y inverse depth
                        
                            
                    else:
                        T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0) # why it is multiplied y inverse depth 

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                # this will be the decomposed new image. multiplied with gaussian in compute losses
                outputs[("color_aug_decompose", frame_id, scale)] = F.grid_sample(
                    inputs[("color_aug_decompose", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color_aug_decompose", frame_id, source_scale)]
        return 
    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) # this should be 0-1
            
            # depth = disp_to_depth_no_scaling(disp) # this should be 0-1

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn": # not used in out code ?

                    axisangle = outputs[("axisangle", 0, frame_id)] # this will be treated as euler
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    if self.opt.use_euler:
                        T = transformation_from_parameters_euler(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0) # we are not using this. why multiplying with mean_inv_depth
                    else:
                        
                        T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                    
                   
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                # shoudl this be color color_aug and not color 
                
                # orignal is color and I changed it to color_aug
                # outputs[("color", frame_id, scale)] = F.grid_sample(
                #     inputs[("color_aug", frame_id, source_scale)],
                #     outputs[("sample", frame_id, scale)],
                #     padding_mode="border", align_corners=True) # chaneg padding_mode="zeros"
                
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color_aug", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True) 

                # original
                # if not self.opt.disable_automasking:
                #     outputs[("color_identity", frame_id, scale)] = \
                #         inputs[("color", frame_id, source_scale)]
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color_aug", frame_id, source_scale)]

                if self.opt.optical_flow:
                    outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                            cam_points, inputs[("K", source_scale)], T)
                    

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = self.frac * ssim_loss + (1-self.frac) * l1_loss # original frac = 0.85

        return reprojection_loss

    def compute_losses_gauss(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        
        # I want to calculate loss after combining the decomposed image with the gauss mask estimated 
        # add gauss mask and the original inputs in the image 
        losses = {}
        total_loss = 0
        gan_loss = 0 
        disc_loss = 0 
        gan_loss_total = 0 
        if self.opt.adversarial_prior:
            # how far is the model from valid examples 
            self.Discriminator.eval()
            output_disc = self.Discriminator(outputs[("depth", 0, 0)]) # this may not be correct because of the scale. check whether we want to miniize disp or depth
            disc_loss = self.criterion_Discriminator(output_disc, self.valid)
            losses["depth_loss/{}".format(0)] = disc_loss
            
            # how far is it from CT
 
        if self.opt.pre_trained_generator:
            
            # Megha
            image = self.gen_transform(inputs[("color_aug", 0, 0)])
            fake_B1 = self.models["pre_trained_generator"](image)
            # fake_B1_norm = 1.0 - Rescale(fake_B1)()
            
            # don't change this 
            depth = disp_to_depth_no_scaling(disp)
            
            _ , fake_disp_scaled = depth_to_disp(fake_B1)
            
            
                
            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]
                # upscale 
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                
                gan_loss = self.si_loss(fake_disp_scaled,disp)
            
                losses["gan_loss/{}".format(scale)] = gan_loss
                
                gan_loss_total= gan_loss_total + gan_loss
        
            # get disp at multiple scales, upscale and then compare with the GAN output.
            
            
            # disp = outputs[("disp", scale)]
            # depth_norm = Rescale(outputs[("depth", 0, 0)])()
            
            # transforms.Normalize(0.5, 0.5)(outputs[("depth", 0, 0)])
            # depth_norm = self.normalize_transform(outputs[("depth", 0, 0)])

            # abs_diff = torch.abs(fake_B1_norm - depth_norm)
            # l1_loss = abs_diff.view(abs_diff.shape[0], abs_diff.shape[1], -1).mean(2).mean(0)
            # losses["gan_loss/{}".format(0)] = l1_loss
            # gan_loss = l1_loss
            # total_loss += l1_loss
                
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)] # this is of different sizes 
            color = inputs[("color_aug_decompose", 0, scale)] # original without gaussian effect
            # target = inputs[("color_aug", 0, source_scale)] # original unchnaged 
            target = inputs[("color_aug_decompose", 0, source_scale)] # original unchnaged 
            outputs.update({("original_aug", 0, scale):inputs[("color_aug", 0, source_scale)]})

            # add gaussian for both the frames, check if its different
            for frame_id in self.opt.frame_ids[1:]:
                # here the size of the ouput should be same as originail image 
                # if self.opts.gauss_number > 1:
                #     pred = outputs[("color_aug_decompose", frame_id, scale)] * inputs[("gaussian_mask1", frame_id, 0)] *inputs[("gaussian_mask2", frame_id, 0)]# add gaussian masks 
                # else:
                #     pred = outputs[("color_aug_decompose", frame_id, scale)] * inputs[("gaussian_mask1", frame_id, 0)]
                
                # pred = outputs[("color_aug_decompose", frame_id, scale)] * (inputs[("gaussian_mask1", frame_id, 0)]/4 + inputs[("gaussian_mask2", frame_id, 0)]/4 + inputs[("gaussian_mask3", frame_id, 0)]/4 + inputs[("gaussian_mask4", frame_id, 0)]/4)
                pred = outputs[("color_aug_decompose", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                
                outputs.update({("color_aug_compose", frame_id, scale):pred})
                

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # This will 0 in case of the similar regions 
            if not self.opt.disable_automasking: # To detect static frames or regions with specularity
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    # pred = inputs[("color_aug", frame_id, source_scale)] # tricky, if same as previous
                    pred = inputs[("color_aug_decompose", frame_id, source_scale)] # tricky, if same as previous
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties, why because this can be zero ?
                # identity_reprojection_loss += torch.randn(
                #     identity_reprojection_loss.shape, device=self.device) * 0.000000001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1) # multiply with mask[mask < 0.3]
                to_optimise = to_optimise*inputs[("gaussian_remove_mask", 0)].squeeze()

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()*(inputs[("gaussian_remove_mask", 0)].squeeze()!=0) # if the loss is in dim 2 or 3 

                # outputs["identity_selection/{}".format(scale)] = (
                #     idxs > identity_reprojection_loss.shape[1] - 1).float() # if the loss is in dim 2 or 3 
                
            loss += to_optimise.mean()
            
            losses["min_loss/{}".format(scale)] = to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # here use the decomposed one because disparity is wrt to that. 
            smooth_loss = get_smooth_loss(norm_disp, color) # color is decomposed_aug

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss + gan_loss_total/self.num_scales * 0.002 + 0.02 * disc_loss
        return losses
         
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        gan_loss = 0 
        disc_loss = 0 
        gan_loss_total = 0 
        if self.opt.pose_prior:
            pose_loss = 0
            for frame_id in [-1, 1]:
                axisangle_trans_5 = torch.cat([outputs[("axisangle", 0, frame_id)][:, 0, :, :2], outputs[("translation", 0, frame_id)][:, 0, :, :]], 2)
                prior_axisangle_trans_5 = inputs[('pose_prior', frame_id)][:,:5][:, None, :]
                pose_loss+=self.pose_criterion(axisangle_trans_5, prior_axisangle_trans_5) # check this
                
        if self.opt.split == "endoSLAM":
            pred_depth = outputs[('disp', 0)]*inputs[("endoslam_mask", 0)][:, 0, :, :][:, None, :, :]
            gt_depth = (inputs[("color_depth", 0, 0)]*inputs[("endoslam_mask", 0)])[:, 0, :, :][:, None, :, :]
            depth_loss_endoslam = self.depth_endoslam_loss(pred_depth, gt_depth)
        
        
        if self.opt.longterm_consistency_loss:        
            long_term_consistency_loss = self.longterm_consistency_criterion(outputs[("eulerTanslation_lonterm")], outputs[("eulerTanslation_serial")])
            losses["long_term_consistency_loss"] = long_term_consistency_loss
            
        if self.opt.adversarial_prior:
            # how far is the model from valid examples 
            self.Discriminator.eval()
            output_disc = self.Discriminator(outputs[("depth", 0, 0)]*inputs[("gauss_mask", 0)])
            disc_loss = self.criterion_Discriminator(output_disc, self.valid)
            losses["depth_loss/{}".format(0)] = disc_loss
            
            # how far is it from CT
            
        # if self.opt.forward_backward_loss:
        #     outputs[("axisangle", 0, frame_id)]
        if self.opt.pose_consistency_loss:
            pose_consistency_loss = 0

            for frame_id in [-1, 1]:
                reverse_euler_ = matrix_to_euler_angles(euler_angles_to_matrix(outputs[("reverse_axisangle", 0, frame_id)][:,0], 'ZYX')[:,:3,:3], 'XYZ')
                current_euler = outputs[("axisangle", 0, frame_id)][:,0]
                reverse_euler = -torch.flip(reverse_euler_, dims=(1,))[:,None,:]
                
                reverse_translation = torch.matmul(euler_angles_to_matrix(outputs[("reverse_axisangle", 0, frame_id)][:,0], 'ZYX')[:,:3,:3], -outputs[("translation", 0, frame_id)][:,0].permute(0, 2, 1)) 
                curr_translation = outputs[("reverse_translation", 0, frame_id)]
                
                curr_dof    = torch.cat([current_euler, curr_translation[:,0, :, :]], 2)
                rev_dof     = torch.cat([reverse_euler, reverse_translation.permute(0, 2, 1)], 2)
                loss_rot    = self.pose_consistency_criterion_rot(curr_dof[:, :, :3], rev_dof[:,:, :3])
                loss_trans  = self.pose_consistency_criterion_trans(curr_dof[:, :, 3:], rev_dof[:,:, 3:])
                pose_consistency_loss+=(loss_rot + loss_trans)
                
                
            pose_consistency_loss = pose_consistency_loss / 2 # becaue two sets of frames
            losses["pose_consistency_loss"] = pose_consistency_loss
                
        if self.opt.pre_trained_generator:
            
            image = self.gen_transform(inputs[("color_aug_original", 0, 0)])
            fake_B1 = self.models["pre_trained_generator"](image)
            # fake_B1_norm = 1.0 - Rescale(fake_B1)()
            
            # _ , fake_disp_scaled = depth_to_disp(fake_B1)
            
            for scale in self.opt.scales:
                # disp = outputs[("disp", scale)]
                
                depth = outputs[("depth", 0, scale)]
                
                # upscale 
                disp = F.interpolate(
                    depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                
                if self.opt.enable_gauss_mask:
                    fake_B1 = fake_B1*inputs[("gauss_mask", 0)]
                    
                gan_loss = self.depth_gan_log_loss(fake_B1,disp)
                # gan_loss = self.si_loss(fake_disp_scaled,disp)
            
                losses["gan_loss/{}".format(scale)] = gan_loss
                
                gan_loss_total = gan_loss_total + gan_loss
        
            # get disp at multiple scales, upscale and then compare with the GAN output.
            
            
            # disp = outputs[("disp", scale)]
            # depth_norm = Rescale(outputs[("depth", 0, 0)])()
            
            # transforms.Normalize(0.5, 0.5)(outputs[("depth", 0, 0)])
            # depth_norm = self.normalize_transform(outputs[("depth", 0, 0)])

            # abs_diff = torch.abs(fake_B1_norm - depth_norm)
            # l1_loss = abs_diff.view(abs_diff.shape[0], abs_diff.shape[1], -1).mean(2).mean(0)
            # losses["gan_loss/{}".format(0)] = l1_loss
            # gan_loss = l1_loss
            # total_loss += l1_loss
                
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            # color = inputs[("color", 0, scale)]
            # target = inputs[("color", 0, source_scale)]
            
            color = inputs[("color_aug", 0, scale)]
            target = inputs[("color_aug", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target)) 

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    # pred = inputs[("color", frame_id, source_scale)]
                    pred = inputs[("color_aug", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                # identity_reprojection_loss += torch.randn(
                #     identity_reprojection_loss.shape, device=self.device) * 0.00000001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float() # wheres is it used ?

                if self.opt.enable_endoMasking:
                    outputs["identity_selection/{}".format(scale)] = outputs["identity_selection/{}".format(scale)]*inputs[("endoslam_mask", 0)][:, 0, :, :]
                    
            # here add the mask weighted mean 
            if self.opt.enable_gauss_mask:
                to_optimise_masked = to_optimise*inputs[("gauss_mask", 0)].squeeze()
                loss += to_optimise_masked.mean()
            elif self.opt.enable_endoMasking:
                to_optimise_masked = to_optimise*inputs[("endoslam_mask", 0)][:, 0, :, :]
                loss += to_optimise_masked.mean()
            else:  
                loss += to_optimise.mean()
            
            losses["min_loss/{}".format(scale)] = to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            
            # if self.opt.enable_gauss_mask:
            #     smooth_loss = get_smooth_loss_gauss_mask(norm_disp, color, inputs[("gauss_mask", scale)])
            # else:
            #     smooth_loss = get_smooth_loss(norm_disp, color)

            smooth_loss = get_smooth_loss(norm_disp, color)
             
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        if self.opt.pose_prior:
            losses["loss"] = total_loss + gan_loss_total/self.num_scales * 0.002 + 0.002 * disc_loss + pose_loss
        else:
            losses["loss"] = total_loss + gan_loss_total/self.num_scales * 0.0002 + 0.002 * disc_loss 
            
        if self.opt.pose_consistency_loss:
            losses["loss"]+=(pose_consistency_loss*self.opt.pose_consistency_weight)
        
        if self.opt.longterm_consistency_loss:
            losses["loss"]+=(losses["long_term_consistency_loss"]*self.opt.longterm_consistency_weight)
            
        if self.opt.split == "endoSLAM":
            losses["depth_mse"] = depth_loss_endoslam
            losses["loss"]+=depth_loss_endoslam
            
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    def log_wand(self, mode, outputs, losses, wand_object, step, character, lr = 0, use_discriminator_loss = False, discriminator_loss = 0,
                 discriminator_response= None, gaussian_decomposition = False, gaussian_response = None):
        # output here is disparity image 
        wand_object.log_data(outputs, losses, mode, character=character, step = step, learning_rate = lr, use_discriminator_loss = use_discriminator_loss, 
                            discriminator_loss= discriminator_loss, discriminator_response= discriminator_response, gaussian_decomposition = gaussian_decomposition, gaussian_response = gaussian_response ) # step can also be epoch 
        
    def log(self, mode, inputs, outputs, losses, disc_reponse = None, disc_loss = 0 , add_discriminator_loss = False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        if add_discriminator_loss:
            writer.add_scalar("disc_loss", disc_loss, self.step)
            writer.add_image("disc_response",disc_reponse, self.step)
            
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        
        if self.opt.adversarial_prior:
            save_path_disc = os.path.join(save_folder, "discriminator.pth")
            torch.save(self.Discriminator.state_dict(), save_path_disc)

            save_path_disc_optim = os.path.join(save_folder, "{}.pth".format("adam_disc"))
            torch.save(self.optimizer_Discriminator.state_dict(), save_path_disc_optim)
            
            
            
    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        # if discrmininator model 
        if self.opt.load_discriminator:
            self.opt.models_to_load.append('discriminator')
            
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")
            
        if self.opt.load_discriminator:
            disc_optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam_disc.pth")
            if os.path.isfile(disc_optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict_disc = torch.load(disc_optimizer_load_path)
                self.optimizer_Discriminator.load_state_dict(optimizer_dict_disc)
            else:
                print("Cannot find discriminator Adam weights so Adam is randomly initialized")
            
        
            
            
