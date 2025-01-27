

import os
import torch
import numpy as np
import yaml

from tqdm import tqdm
import sys
sys.path.append('../')
from datamodule import datamodule


def subsample_baselineDataset_10perclass(train_loader, valid_loader):
    """
    Subsample the dataset to have 10 samples per class
    """

    subsampled_data = []
    for i in tqdm(range(0,10)):
        num_in_class = 0
        for batch in train_loader:
            _, _, target = batch
            if torch.argmax(target.squeeze()) == i:
                subsampled_data.append(batch)
                num_in_class += 1
            if num_in_class == 2:
                break
        for batch in valid_loader:
            _, _, target = batch
            if torch.argmax(target.squeeze()) == i:
                subsampled_data.append(batch)
                num_in_class += 1
            if num_in_class == 3:
                break

    return subsampled_data



if __name__ == "__main__":

    # Load the config file
    config = yaml.load(open('../../config_coop.yaml', 'r'), Loader=yaml.FullLoader)
    path_root = '../../'
    config['paths']['path_root'] = path_root

    datamodule = datamodule.select_data(config)
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()

    subsampled_data = subsample_baselineDataset_10perclass(train_loader, valid_loader)
    from IPython import embed; embed()


