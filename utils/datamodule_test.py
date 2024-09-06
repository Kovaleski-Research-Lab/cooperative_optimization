import sys
import yaml
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../../')

import src.datamodule as datamodule

if __name__ == "__main__":

    params = yaml.load(open('../config.yaml', 'r'), Loader=yaml.FullLoader)
    params['batch_size'] = 1
    params['which'] = 'MNIST'
    params['paths']['path_root'] = '../'
    
    # Initialize the datamodule
    dm = datamodule.select_data(params)
    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()

    # Get a sample from the train loader to get the image shape
    batch = next(iter(train_loader))
    image = batch[0].numpy().squeeze().real
    w,h = image.shape

    # Create a figure for displaying images
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Sample {}'.format(0))
    
    # Loop through some images and display them
    for i,batch in enumerate(train_loader):
        image = batch[0].numpy().real
        im.set_data(image.squeeze())
        ax.set_title('Sample {}'.format(i))
        plt.pause(0.1)

