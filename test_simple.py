from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import torch
from torchvision import transforms, datasets
from options_eval import MonodepthEvalOptions

import networks
from layers import disp_to_depth, transformation_from_parameters

import csv

print("Integers written to CSV file successfully.")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        default = 'data/dataset_14/keyframe_1' )
    parser.add_argument('--model_path', type=str,
                        help='path to the test model', default ='test/weights_19') #models_pretrained/Model_MIA")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    parser.add_argument("--pose_model_type",
                        type=str,
                        help="normal or shared",
                        default="separate_resnet",
                        choices=["posecnn", "separate_resnet", "shared"])
    
    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = args.model_path

    pose_prediction = False
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    pose_ecoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")
    
    
    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False) 
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained 
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}

    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print(" Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    
    # # pose # loading
    pose_encoder = networks.ResnetEncoder(18,num_input_images=2, pretrained = 1)
    loaded_dict_pose = torch.load(pose_ecoder_path, map_location=device)
    pose_encoder.load_state_dict(loaded_dict_pose)
    pose_encoder.to(device)
    pose_encoder.eval()
    
    pose_encoder = networks.ResnetEncoder(18,num_input_images=2, pretrained = 1)
    loaded_dict_pose = torch.load(pose_ecoder_path, map_location=device)
    pose_encoder.load_state_dict(loaded_dict_pose)
    pose_encoder.to(device)
    pose_encoder.eval()
    
    pose_decoder_model = networks.PoseDecoder(num_ch_enc=pose_encoder.num_ch_enc, num_input_features = 1, num_frames_to_predict_for=2) 
    pose_decoder_dict = torch.load(pose_decoder_path, map_location=device)
    pose_decoder_model.load_state_dict(pose_decoder_dict)
    pose_decoder_model.to(device)
    pose_decoder_model.eval()
    
    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths): # should be sequntial

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')

            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

            if idx < len(paths)-1:
                input_image1 = pil.open(paths[idx+1]).convert('RGB')
                input_image1 = input_image1.resize((feed_width, feed_height), pil.LANCZOS)
                input_image1 = transforms.ToTensor()(input_image1).unsqueeze(0).to(device)
                inputs_all = [input_image, input_image1]
            
            # PREDICTION
            # input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height * 2, original_width * 2), mode="bilinear", align_corners=False)
            
            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 150)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)

            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax) # 归一化到0-1
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma') # colormap
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}.jpeg".format(output_name))
            im.save(name_dest_im, quality=95)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

            # pose prediction 
            if pose_prediction:
                
                pose_inputs = [pose_encoder(torch.cat(inputs_all, 1))] # cat imaege 1 and img 2
                axisangle, translation = pose_decoder_model(pose_inputs) # why is it giving two value and why is it axis angle and not euler or quaternion
                cam_T_cam = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).to('cpu')
                
                row = []
                with open('rot_trans.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(np.hstack((axisangle[:, 0].squeeze().cpu().numpy(), translation[:, 0].squeeze().cpu().numpy())))
                
                with open('transform.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in cam_T_cam.squeeze().numpy():
                        writer.writerow(row)
                
    print('->p Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
