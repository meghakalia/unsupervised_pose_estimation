


import wandb
import torchvision
from torchvision import transforms
import torch

import PIL.Image as pil
from torchvision import transforms
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from typing import overload

class wandb_logging:
    
   
    def __init__(self, options, models = None):
        wandb.login()
        
        self.opts = options

        self.config = vars(options)
        self.config.update({'name':"phantom_Dataset_vanilla_hyperparameter_search"})
        self.config.update({'align_corner':"True"})
        self.config.update({'augmentation':"True"})
        
        
        # self.config = dict(
        # height = self.opts.height,
        # width = self.opts.width,
        # epochs=self.opts.num_epochs,
        # batch_size=self.opts.batch_size,
        # learning_rate=self.opts.learning_rate,
        # dataset="phantom_Dataset_vanilla_hyperparameter_search",
        # frame_ids = self.opts.frame_ids,
        # scales = self.opts.scales,
        # augmentation = "True",
        # align_corner="True")
        
        self.resize = transforms.Resize((self.config['height'], self.config['width']))
        
        wandb.init(project="drop_out_test", config=self.config, dir = 'data/logs')
        
        self.save_colored_depth = False
        
        if models:
            self.models = []
            for model in models:
                self.models.append(model)
                wandb.watch(self.models[-1], log_freq=1000, log='all') # default is 1000, it makes the model very slow

        return 

    def startSweep(self, sweep_configuration, project_name, function_to_run, count):
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

        # wandb.agent(sweep_id, function=main, count=10)       
        wandb.agent(sweep_id, function=function_to_run, count=count)
        
    def finishWandb(self):
        wandb.finish()
        return
    
    def get_config(self):
        return self.config
        
    def log_data_stage2(self):
        return 
    
    def log_gradients(self, names=["gradient_norm"]):
        count = 0 
        for model in self.models:    
            # not correct throwing error 
            wandb.log({names[count]: model.named_parameters()})
            if len(names) > 1: 
                count+=1
        
    def log_lr(self, lr):
        wandb.log({'lr': lr})
    
    def log_data(self, outputs, losses, mode, step=1, character="registration", stage=1, learning_rate = 0):
       
        # log losses 
        # k = [key for key, value in losses.items()]
        
        for l, v in losses.items():
            wandb.log({"{}_{}".format(mode, l):v, 'custom_step':step})
            
        wandb.log({"lr":learning_rate, 'custom_step':step})
        
        # log images 
        if outputs.get('trajectory', 0):
            wandb.log({"{}_{}".format(mode, 'trajectory'):wandb.Image(outputs['trajectory'], caption = ''),'custom_step':step})  


        # for j in range(min(4, self.config['batch_size'])):  # write a maxmimum of four images
        if character != "trajectory":
            for s in self.config['scales']:
                image_list = []
                caption_list = []
                image_list_depth = []
                image_list_automask = []
                image_list_color = []
                image_list_pred_color = []
                
                for frame_id in self.config['frame_ids'][1:]: # what is logged here 

                # image_list.append(inputs[("color", 0, 0)][j].data)
                
                    # list_1 = outputs[("registration", s, frame_id)][:4,:,:]
                    if character=="registration":
                        image_list.append(outputs[("registration", s, frame_id)][:4,:,:])
                        
                if character=="disp":
                    
                    # colormap depth 
                    if self.save_colored_depth:
                        disp = outputs[("disp", s)][:1,:,:]
                        disp_resized_np = disp.squeeze().cpu().numpy()
                        vmax = np.percentile(disp_resized_np, 95)

                        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax) 
                        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma') # colormap
                        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

                        im = pil.fromarray(colormapped_im)
                        
                        image_list.append(im)
                        
                        wandb.log({"{}_{}".format(mode, s):wandb.Image(im, caption = ''),'custom_step':step})  
                    else:
                        image_list.append(outputs[("disp", s)][:4,:,:]) # first 4 images of the images
                        
                        if ("depth", 0, s) in outputs:
                            image_list_depth.append(outputs[("depth", 0, s)][:4,:,:])
                        
                        if "identity_selection/{}".format(s) in outputs:
                            image_list_automask.append(outputs["identity_selection/{}".format(s)][:4,:,:]*255)
                        
                        if ("color", frame_id, s) in outputs:
                            image_list_pred_color.append(outputs[("color", frame_id, s)][:4,:,:])
                
                
                if not self.save_colored_depth:
                    c = torch.concat(image_list, 0)
                    self.log_image_grid(mode = mode, image_list = c, scale = s, caption = ' ', character = ''.join((character,"{}".format(s))), step = step)
                    
                    c_depth = torch.concat(image_list_depth, 0)
                    self.log_image_grid(mode = mode, image_list = c_depth, scale = s, caption = 'depth_', character = ''.join((character,"{}".format(s))), step = step)
                    
                    c_automask = torch.concat(image_list_automask, 0)
                    c_automask = c_automask[:, None, :, :]
                    self.log_image_grid(mode = mode, image_list = c_automask, scale = s, caption = 'automask_ ', character = ''.join((character,"{}".format(s))), step = step)
                    
                    c_pred_color = torch.concat(image_list_pred_color, 0)
                    self.log_image_grid(mode = mode, image_list = c_pred_color, scale = s, caption = 'pred_color ', character = ''.join((character,"{}".format(s))), step = step)
                # row = len(image_list), 
                
                
            # for s in self.config['scales']:
                # make grid here, with original input images
                # self.log_single_image_fromtensor(mode, outputs[("registration", s, frame_id)][j].data, "registration{}_{}_{}".format(j, frame_id, s))
                # image_list.append(self.resize(outputs[("registration", s, frame_id)][j]).data) # resize # slow 
                
                                    
                # caption_list.append("{}".format(frame_id))

            # test once 
            # self.log_image_grid(mode, image_list, s, caption_list, row = len(image_list), character = "registration{}_{}".format(j, frame_id))
                    
                # wandb.log({"{}_{}}".format(mode, stage):wandb.Image(outputs[("registration", s, frame_id)][j].data,caption="registration_{}_{}/{}".format(frame_id, s, j))})
                    
    
    def save_model(self, path):
        return 
    
    def log_image_grid(self, mode, image_list, scale, caption, character = '', step= 1):
        img_grid = torchvision.utils.make_grid(image_list)
        
        npimg = img_grid.permute(1, 2, 0).cpu().numpy()
        self.log_single_image(''.join((mode,str(scale),caption)), image = npimg, caption = "{}_{}_{}_{}".format(character, mode, scale, ''.join(caption)), step=step)
        return 
    
    def log_single_image(self, dict_key, step, image, caption=''):
        wandb.log({"{}".format(dict_key):wandb.Image(image, caption = caption),'custom_step':step})  
        return 
    
    def log_single_image_fromtensor(self, dict_key, image, caption=''):
        
        npimg = image.cpu().numpy()
        if image[0]==3:
            npimg = image.permute(1, 2, 0).cpu().numpy()
            
        wandb.log({"{}".format(dict_key):wandb.Image(npimg, caption = caption)})  
        return 

    

        