# Going to evaluate the classifier (trained on in-focus bench images) on 
# out-of-focus bench images. Going to sweep the focal length of the lens
# between +- 10% of the ideal focal length. At each focal length, we run the 
# dataset and save model predictions / ideal targets.
# The goal here is to find the 'line in the sand' on where the classifier
# starts to fail.


import os
import numpy as np
import torch
import yaml
import sys

sys.path.append('../')
from models.models import CooperativeOpticalModelRemote
from datamodule import datamodule
from pytorch_lightning import seed_everything



if __name__ == "__main__":

    config = yaml.load('../../config_coop.yaml', Loader=yaml.FullLoader)
    seed_everything(config['seed'][1], workers=True)
    path_root = '../../'
    config['paths']['path_root'] = path_root

    # Ideal focal length
    ideal_focal_length = 287.75

    # Focal length sweep
    focal_length_sweep = np.linspace(ideal_focal_length*0.9, ideal_focal_length*1.1, 20)

    # For each focal length
    for focal_length in focal_length_sweep:
        config['modulators'][1]['focal_length'] = focal_length
        model = CooperativeOpticalModelRemote(config)
        model.eval()
        model.cuda()

        # Initialize the datamodule
        dm = datamodule.select_data(config)
        dm.setup()
        train_loader = dm.train_dataloader()
        valid_loader = dm.val_dataloader()

        # Run the datamodule

