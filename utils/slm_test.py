import os
import sys
import yaml
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21

import src.datamodule as datamodule

if __name__ == "__main__":

    params = yaml.load(open('../config.yaml', 'r'), Loader=yaml.FullLoader)
    params['batch_size'] = 1
    params['which'] = 'MNIST'
    
    # Initialize the datamodule
    dm = datamodule.select_data(params)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()

    slm = holoeye_pluto21.HoloeyePluto(host = '10.10.80.1')

    # Loop through some images, upload to SLM - no way to display them here,
    # just making sure the SLM can handle the images
    # The process is a little wild here.
    for i,batch in enumerate(train_loader):
        image = batch[1].numpy()
        image = image.squeeze()
        image = Image.fromarray(image)
        image.save('temp.png')
        slm.send_scp('temp.png')
        slm.update(filename = 'temp.png', options = ['shf=1', 'wl=520','-fx'], wait=1.0)
        if i > 3:
            break

    # Clean up the saved image
    os.remove('temp.png')


