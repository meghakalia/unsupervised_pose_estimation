from __future__ import absolute_import, division, print_function

import os
import torch
import networks
import numpy as np

from torch.utils.data import DataLoader
from layers import transformation_from_parameters
from utils import readlines
from options_eval import MonodepthEvalOptions
from datasets import LungRAWDataset

import matplotlib.pyplot as plt

def sample_filenames_frequency(filenames, sampling_frequency):
    outputfilenames = []
    count = 0
    outputfilenames.append(filenames[0])
    for file in filenames:
        
        if count == sampling_frequency:
            outputfilenames.append(file)
            count = 0 

        count+=1
    
    return outputfilenames

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        rs.append(cam_to_world[:3, :3])
    return rs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]

def compute_scale(gtruth, pred):

    # Optimize the scaling factor
    gtruth = gtruth[:len(pred), :, :]
    scale = np.sum(gtruth[:, :3, 3] * pred[:, :3, 3]) / np.sum(pred[:, :3, 3] ** 2)
    return scale

def plotTrajectory(pred_poses, gt_local_poses, save_fig = False, name = 0):
    our_local_poses = pred_poses
    # gt_local_poses_absolute = loadGTposes(our_path_gt)
    gt_local_poses = gt_local_poses[:len(pred_poses), :, :]
    dump_our = np.array(dump(our_local_poses))
    dump_gt = np.array(dump(gt_local_poses))

    scale_our = dump_our * compute_scale(dump_gt, dump_our)
    
    num = len(gt_local_poses) # shoudl be array
    points_our = []
    points_gt = []
    origin = np.array([[0], [0], [0], [1]])

    for i in range(0, num):
        point_gt = np.dot(dump_gt[i], origin)
        point_our = np.dot(scale_our[i], origin)

        points_our.append(point_our)
        points_gt.append(point_gt)

    points_our  = np.array(points_our)
    points_gt   = np.array(points_gt)

    # new a figure and set it into 3d
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    # set figure information
    # ax.set_title("3D_Curve")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    # draw the figure, the color is r = read
    figure1, = ax.plot(points_gt[:, 0, 0], points_gt[:, 1, 0], points_gt[:, 2, 0], c='b', linewidth=1.6)
    figure2, = ax.plot(points_our[:, 0, 0], points_our[:, 1, 0], points_our[:, 2, 0], c='g', linewidth=1.6)

    if save_fig:
        plt.savefig('{}.png'.format(name),dpi=600)
    
    return plt
    # plt.show()
    
    
def dump(source_to_target_transformations):
    Ms = []
    cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        Ms.append(cam_to_world)
    return Ms

def evaluate(opt):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    # filenames = readlines(
    #     os.path.join(os.path.dirname(__file__), "splits", "scared",
    #                  "test_files_phantom14.txt"))[0:50]
    
    # num = 11
    for num in range(1, 15):
        filenames_1 = readlines(
            os.path.join(os.path.dirname(__file__), "splits", "endovis",
                        "test_files_phantom_{}.txt".format(num)))
        filenames = sample_filenames_frequency(filenames_1, sampling_frequency = 3)
        
        # filenames = filenames_1
        # dataset = SCAREDRAWDataset(opt.data_path, filenames, opt.height, opt.width,
        #                            [0, 1], 4, is_train=False)
        # dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
        #                     num_workers=opt.num_workers, pin_memory=True, drop_last=False)
        
        dataset = LungRAWDataset(
                opt.data_path, filenames, opt.height, opt.width,
                [0, 1], 4, is_train=False, len_ct_depth_data = len(filenames), data_augment = False, sampling_frequency = 3)
        
        dataloader = DataLoader(dataset, 1, shuffle=False, drop_last=False, pin_memory=True)
        
        # check time that dataloader takes to load the samples
        
        # dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,pin_memory=False, drop_last=False)
        
        pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
        pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

        pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))

        pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))

        pose_encoder.cuda()
        pose_encoder.eval()
        pose_decoder.cuda()
        pose_decoder.eval()

        pred_poses = []

        print("-> Computing pose predictions")

        opt.frame_ids = [0, 1]  # pose network only takes two frames as input

        count = 0 
        axisangle_ = []
        translation_ = []
        with torch.no_grad():
            for inputs in dataloader:
                
                # count = count + 1
                # print(count)
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()

                all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)

                features = [pose_encoder(all_color_aug)]
                axisangle, translation = pose_decoder(features)
                # axisangle_.append(axisangle[:, 0].cpu().numpy())
                # translation_.append(translation[:, 0].cpu().numpy())

                pred_poses.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                
        # if want to save
        # np.savez('pose_prediction_{}.npz'.format(num), pred_poses)
        # np.savez('axisangle_{}.npz'.format(num), axisangle_)
        # np.savez('translation_{}.npz'.format(num), translation_)
        pred_poses = np.concatenate(pred_poses)

        gt_path = os.path.join(os.path.dirname(__file__), "splits", "endovis", "gt_poses_phantom_{}.npz".format(num))
        gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

        plotTrajectory(pred_poses, gt_local_poses, True, name = num)
    # ates = []
    # res = []
    # # num_frames = gt_local_poses.shape[0]
    # num_frames = pred_poses.shape[0] - 3
    # track_length = 5
    # count = 0 
    # for i in range(0, num_frames - 1):
    #     local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
    #     gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
    #     local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
    #     gt_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))
    #     # if i + track_length - 1 > 50:
    #     #     print('here')
    #     # print(i + track_length - 1)

    #     ates.append(compute_ate(gt_local_xyzs, local_xyzs))
    #     res.append(compute_re(local_rs, gt_rs))
        
    # print("\n   Trajectory error: {:0.4f}, std: {:0.4f}\n".format(np.mean(ates), np.std(ates)))
    # print("\n   Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res), np.std(res)))

    # get the error
    # save the image 
    # plotTrajectory(pred_poses, gt_local_poses, True, name = num)


if __name__ == "__main__":
    options = MonodepthEvalOptions()
    evaluate(options.parse())
