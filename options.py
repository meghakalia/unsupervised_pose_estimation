from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

# important parameters 
# range of depth map
# disp 
# validation metric
# sampling frequency


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                              #    default=os.path.join(file_dir, "data"))
                              default=os.path.join(file_dir, "data"))
                                   # default=os.path.join(file_dir, "data_porcine"))
        
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="/code/data/models_disc_prior_logging")
        self.parser.add_argument("--gauss_number",
                                   type=int,
                                   help="gaussin number in gauss decomposition",
                                   default=2)
        self.parser.add_argument("--gauss_lr",
                                   type=float,
                                   help="gaussin number in gauss decomposition",
                                   default=10e-8)
        self.parser.add_argument("--gauss_scheduler_step_size",
                                   type=int,
                                   help="gaussin number in gauss decomposition",
                                   default=5)
        
        self.parser.add_argument("--gauss_pretrained_model_path",
                                   type=str,
                                   help="gaussian pretrained model path",
                                   default='')
        
        self.parser.add_argument("--write_split_file",
                                 help="if set, will do the train-val split and write in a file",
                                 action="store_true")# false
        
        self.parser.add_argument("--gaussian_correction",
                                 help="if set, will do enable gaussian correction",
                                 action="store_true")# false

        # TRAINING options
        self.parser.add_argument("--use_euler",
                                 help="whether euler or axis angle",
                                 action="store_false")
   
        
        self.parser.add_argument("--pre_trained_generator",
                                 help="whether pretrained gan is on or off ",
                                 action="store_false")
        
        self.parser.add_argument("--enable_gauss_mask",
                                 help="weighing the loss with gauss mask",
                                 action="store_false")
        
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="log_loss_gan_multi_scale_consistency_losses")
        
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["endovis", "eigen_zhou", "eigen_full", "odom", "benchmark", "endoSLAM"],
                                 default="endovis")
        
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="endovis",
                                 choices=["endovis", "kitti", "kitti_odom", "kitti_depth", "kitti_test", "vnb", "porcine", "endoSLAM"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192) # for scared data 960//5, default=192, for scared : 256x320
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=192) # for scared data 1280//5, default=192
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)#1e-4
        self.parser.add_argument("--position_smoothness",
                                 type=float,
                                 help="registration smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--consistency_constraint",
                                 type=float,
                                 help="consistency constraint weight",
                                 default=0.01)
        
        
        
        self.parser.add_argument("--epipolar_constraint",
                                 type=float,
                                 help="epipolar constraint weight",
                                 default=0.01)
        self.parser.add_argument("--geometry_constraint",
                                 type=float,
                                 help="geometry constraint weight",
                                 default=0.01)
        self.parser.add_argument("--transform_constraint",
                                 type=float,
                                 help="transform constraint weight",
                                 default=0.01)
        self.parser.add_argument("--transform_smoothness",
                                 type=float,
                                 help="transform smoothness weight",
                                 default=0.01)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.001)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=1.0) # 150.0
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        
        self.parser.add_argument("--eval_pose_trajectory",
                                 help="this will evaluate the model performance on trajectory",
                                 action="store_true")
        
        self.parser.add_argument("--tra_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "data"))

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)
        
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-6) # initial was 1e-4
        
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        
        self.parser.add_argument("--optical_flow",
                                 help="optical flow",
                                 action="store_true")
        
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        
        self.parser.add_argument("--enable_endoMasking",
                        help="if set, doesn't do auto-masking",
                        action="store_true")
         
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--pose_consistency_loss",
                                 action="store_false",
                                 help="true or false")
        
        self.parser.add_argument("--pose_consistency_weight",
                                 type=float,
                                 help="loss weight",
                                 default=0.001)
        
        self.parser.add_argument("--longterm_consistency_loss",
                                 action="store_false",
                                 help="true or false")
        
        self.parser.add_argument("--longterm_consistency_weight",
                                 type=float,
                                 help="loss weight",
                                 default=0.001)
        
        self.parser.add_argument("--pose_prior",
                                 action="store_true",
                                 help="true or false")
        
        self.parser.add_argument("--load_discriminator",
                                 action="store_true",
                                 help="true or false")
        
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 default = None, 
                                 # default = '/code/data2/models_disc_prior_logging/pose_consistency_working_0.0001/models/weights_19', 
                              #    default  = '/code/data/models_depth_scaled/mdp/models/weights_9',
                                 # default  = 'data_gan_depth_to_disp_/mdp/models/weights_19',
                                 # default = '/code/code/4_batch_4_multigaussian_gauss_sum_2_singleGaussaNetwork_recon_pretrained_trainable_dataaug_True_gauss_num_1_batchnorm_True_ssim_l1_0.65_sigma_network_gauss_combinationTrue_same_gausskernel_False_separatemeanstd_True/models/weights_23',
                                 help="name of model to load")
        
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 # default = None)
                                 # default = ["decompose", 'sigma_combined', 'gaussian1'] )
                                 default=["pose_encoder", "pose", "depth", "encoder"])
                              #    default=["position_encoder", "position"])

        # LOGGING options
        self.parser.add_argument("--sampling_frequency",
                                 type=int,
                                 help="sampling frequency of the frames, can be [1,2,3,4]",
                                 default=1)
        
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=100)
        
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        
        self.parser.add_argument("--wandb_sweep",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="endovis",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "endovis"],
                                 help="which split to run eval on")
        
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        
        self.parser.add_argument("--multiscale_adversarial_prior",
                                 help="whether we want to include the prior unity CT labels for depth",
                                 action="store_true") 
        
        self.parser.add_argument("--adversarial_prior",
                                 help="whether we want to include the prior unity CT labels for depth",
                                 action="store_true")
        
        self.parser.add_argument("--discriminator_lr", type=float, default=0.0002, help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
