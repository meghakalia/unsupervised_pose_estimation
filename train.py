# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.



from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

import wandb_logging
import wandb
import numpy as np

options = MonodepthOptions()
opts = options.parse()

if opts.wandb_sweep: 
    wanb_obj = wandb_logging.wandb_logging(opts)
    wandb_config = wanb_obj.get_config()
    
def main():
    
    # learning_rate_opt = [2, 3, 4, 5, 6, 7 , 8]
    # learning_rate_opt = np.random.permutation(7) + 2
    # for i in range(7):
    # #    idx = np.random.randint(0,6)
    #    learn_rate = 10**(-float(learning_rate_opt[i]))
    learn_rate = 10**(-4)
    frequency = 3
    # for frequency in [1, 2, 3]:
    trainer = Trainer(opts, lr = learn_rate, sampling=frequency)
    trainer.train()
    
if opts.wandb_sweep: 
    sweep_configuration = {
            "method": "random",
            "metric": {"goal": "minimize", "name": "train2_loss"},
            "parameters": {
                "learning_rate": {"max": 1e-3, "min": 1e-8},
                "sampling_frequency": {"values": [1, 2, 3, 4]},
            },
        }
    
    wanb_obj.startSweep(sweep_configuration=sweep_configuration, project_name="my-first-sweep", function_to_run =  main, count = 5)
    
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
    # wandb.agent(sweep_id, function=main, count = 5)
    
    # wandb.startSweep(wandb_config, 'wandb_sweep_first', main, 10)
    
else:
    main()
    
# if __name__ == "__main__":
#     trainer = Trainer(opts)
#     trainer.train()
    
    
    
