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

from datasets import LungRAWDataset

from torchvision.utils import save_image

import wandb_logging

import torchvision.transforms as transforms

from evaluate_pose import plotTrajectory, dump_xyz, dump_r, compute_ate, compute_re
# from torchviz import make_dot

class Trainer:
    def __init__(self, options, lr = 1e-6, sampling = 1, wandb_sweep = False, wandb_config = '', wandb_obj = None):
        
        self.opt = options
        print('learning rate {} sampling frequency : {}'.format(lr, sampling))
        
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
            self.sampling_frequency = sampling
            self.learning_rate = lr
            
            self.opt.learning_rate = lr
            self.opt.sampling_frequency = sampling
            self.wanb_obj = wandb_logging.wandb_logging(self.opt, experiment_name = 'gaussTrain_{}_disc_prior_{}'.format(False, 'patchGAN'))

        # set the manually the hyperparamters you want to optimize using sampling_frequency and learning rate
        
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

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

        
        if self.opt.adversarial_prior: 
            
            self.disc_response = {}
            # Define model 
            # input_shape = (1, self.opt.width, self.opt.height) # this we will have to check
            self.criterion_Discriminator = torch.nn.BCEWithLogitsLoss()
            
            if self.opt.multiscale_adversarial_prior: 
                
                for i in range(len(self.opt.scales)):
                    input_shape[i] = (1, self.opt.width//2, self.opt.height//2)
                    # self.Discriminator[i] = networks.Discriminator(input_shape[i])
                    self.Discriminator[i] = networks.DiscriminatorUnet(input_shape[i])
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
           
        if self.opt.gaussian_correction:
            
            self.resize_transform = {}
            for s in self.opt.scales:
                self.resize_transform[s] = Resize((192// 2 ** s,192// 2 ** s))
                
            
            self.gauss_parameters_to_train = []
            self.models['decompose'] = networks.UNet(3, 3)
            
            self.gaussian_mask1 = []
            self.gaussian_mask2 = []
            self.gauss_reconstructed = []
            
            for g in range(1, self.opt.gauss_number+1):
                self.models['sigma{}'.format(g)] = networks.FCN(output_size = 4) # 4 for each of std x, std y, mean x , mean y
                self.models['gaussian{}'.format(g)] = networks.GaussianLayer(self.opt.height)
                
                self.models['sigma{}'.format(g)].to(self.device)
                self.models['gaussian{}'.format(g)].to(self.device)
                
            self.models['decompose'].to(self.device)

            self.gauss_parameters_to_train += list(self.models["decompose"].parameters())
            
            for g in range(1, self.opt.gauss_number+1):
                self.gauss_parameters_to_train += list(self.models['sigma{}'.format(g)].parameters())
                self.gauss_parameters_to_train += list(self.models['gaussian{}'.format(g)].parameters())
            
            self.gauss_model_optimizer = optim.Adam(self.gauss_parameters_to_train, self.opt.gauss_lr)
            self.gauss_model_lr_scheduler = optim.lr_scheduler.StepLR(self.gauss_model_optimizer, self.opt.gauss_scheduler_step_size, 0.1)
            
            self.models['decompose'].eval()
            self.models['sigma1'].eval()
            self.models['sigma1'].eval()
            self.models['gaussian1'].eval()
            self.models['gaussian2'].eval()
            
                # load model
                
                
               
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

        datasets_dict = {"endovis": datasets.LungRAWDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files_phantom_sampling_freq_5.txt")
        
        train_filenames = readlines(fpath.format("train")) # exclude frame accordingly
        val_filenames = readlines(fpath.format("val"))

        # train_filenames = readlines(fpath.format("train"))[self.sampling_frequency+2:-self.sampling_frequency-6] # exclude frame accordingly
        # val_filenames = readlines(fpath.format("val"))[self.sampling_frequency+2:-self.sampling_frequency-6]
        
        
        img_ext = '.png'

        
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, adversarial_prior = self.opt.adversarial_prior, len_ct_depth_data = 2271, sampling_frequency = self.sampling_frequency )
        
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            drop_last=True)
        
        # self.train_loader = DataLoader(
        #     train_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, adversarial_prior = False, len_ct_depth_data = 0, sampling_frequency = 2)
        
        # self.val_loader = DataLoader(
        #     val_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True, drop_last=True)
        
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
                val_trajectory_14_dataset, self.opt.batch_size, True, drop_last=True)
            
            self.val_traj_1_loader = DataLoader(
                val_trajectory_1_dataset, self.opt.batch_size, True, drop_last=True)
            
            # self.val_iter_1_traj = iter(self.val_traj_1_loader)
            # self.val_iter_14_traj = iter(self.val_traj_14_loader)

            
            # gt poses 
            self.gt_local_poses_14 = np.load(fpath_gt.format("14"), fix_imports=True, encoding='latin1')["data"]
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
                self.log_wand("val2", traj_outputs, traj_losses, self.wanb_obj, step = self.epoch, character="trajectory")
            
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
        self.wanb_obj.finishWandb()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        d_loss = 0
        num_run = 0 
        for batch_idx, inputs in enumerate(self.train_loader):

            num_run+=1
            before_op_time = time.time()

            # process and update inputs here 
            if self.opt.gaussian_correction:
                
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                
                gaussian_reponse = {'gaussian_mask1':[], 'gaussian_mask2':[], 'reconstructed':[], 'decomposed':[], 'original':[]}
                for frame_id in self.opt.frame_ids:
                    features                = self.models["decompose"](inputs["color_aug", frame_id, 0])
                    decomposed    = features[1]
                    
                   
                
                    sigma_out1               = self.models['sigma1'](features[0]) # will spit out 5, 1 gaussian std 
                    gaussian_mask1           = self.models["gaussian1"](sigma_out1)
                    
                    sigma_out2               = self.models['sigma2'](features[0]) # will spit out 5, 1 gaussian std 
                    gaussian_mask2           = self.models["gaussian2"](sigma_out2)
                    
                    # check this, this looks like as if this should be addition
                    re_composed = decomposed * gaussian_mask1[0] * gaussian_mask2[0]
                    
                    gaussian_reponse['original'].append(inputs["color_aug", frame_id, 0][0, :, :, :]) 
                    # update the images
                    for s in self.opt.scales:
                        inputs["color_aug", frame_id, s] = self.resize_transform[s](decomposed)

                    gaussian_reponse['gaussian_mask1'].append(gaussian_mask1[0][0, :, :, :]) 
                    gaussian_reponse['gaussian_mask2'].append(gaussian_mask2[0][0, :, :, :]) 
                    gaussian_reponse['reconstructed'].append(re_composed[0, :, :, :]) 
                    gaussian_reponse['decomposed'].append(decomposed[0, :, :, :]) 
                    
                    
                    
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            
            
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
                self.set_train()
                # self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
        
        print('disc_loss_{}'.format(d_loss/num_run))
        # self.model_lr_scheduler.step()

    def discriminator_train_step(self, inputs, outputs):
        
        # for key, ipt in inputs.items():
        #     inputs[key] = ipt.to(self.device)
            
        # backpropagate through discriminator
        self.Discriminator.train()
        self.optimizer_Discriminator.zero_grad()

        ct_loss_disc = self.Discriminator(inputs[('ct_prior', 0)])
        loss_real = self.criterion_Discriminator(ct_loss_disc, self.valid)
        
        self.disc_response[('disc_response_ct')] = ct_loss_disc
        
        loss_fake = 0 
        for scale in self.opt.scales:
            depth_disc_res = self.Discriminator(outputs[("depth", 0, scale)].detach())
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

                # transforms.ToPILImage()(inputs[("color", 1, 0)].cpu().squeeze()).save('trainer_2_1.png')
                
                all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)

                # save image here all color_aug channel 0 and channel 1
                features = [self.models["pose_encoder"](all_color_aug)]
                axisangle, translation = self.models["pose"](features)

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
    
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

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
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))] # [it is : -1, 0, 1]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation              = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)]      = axisangle
                    outputs[("translation", 0, f_i)]    = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

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
            outputs, losses = self.process_batch(inputs)
            
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

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # upscaling disparities
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) # this should be 0-1
            
            depth = disp_to_depth_no_scaling(disp) # this should be 0-1

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

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                # backprojecting at zero scale only
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                # outputs[("color", frame_id, scale)] = F.grid_sample(
                #     inputs[("color", frame_id, source_scale)],
                #     outputs[("sample", frame_id, scale)],
                #     padding_mode="border", align_corners=True)
                
                outputs[("color_aug", frame_id, scale)] = F.grid_sample(
                    inputs[("color_aug", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                # if not self.opt.disable_automasking:
                #     outputs[("color_identity", frame_id, scale)] = \
                #         inputs[("color", frame_id, source_scale)]
                        
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color_aug", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
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
            
            image = self.gen_transform(inputs[("color", 0, 0)])
            fake_B1 = self.models["pre_trained_generator"](image)
            # fake_B1_norm = 1.0 - Rescale(fake_B1)()
            
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

            disp = outputs[("disp", scale)]
            # color = inputs[("color", 0, scale)]
            # target = inputs[("color", 0, source_scale)]
            
            color = inputs[("color_aug", 0, scale)]
            target = inputs[("color_aug", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color_aug", frame_id, scale)]
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
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

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

            loss += to_optimise.mean()
            
            losses["min_loss/{}".format(scale)] = to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss + gan_loss_total/self.num_scales * 0.002 + 0.02 * disc_loss
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
            
        
            
            
