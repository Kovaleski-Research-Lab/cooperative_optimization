
import os
import sys
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu

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
    camera = thorlabs_cc215mu.Thorlabs_CC215MU()

    # Get an image from the camera to set the figure size
    image = None
    while image is None:
        image = camera.get_image(pil_image = False, eight_bit = False)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Frame {}'.format(0))

    # Loop through images, upload to SLM, get the image back from the camera
    for i,batch in enumerate(train_loader):
        image = batch[1].numpy()
        image = image.squeeze()
        image = Image.fromarray(image)
        image.save('temp.png')
        slm.send_scp('temp.png')
        slm.update(filename = 'temp.png', options = ['shf=1', 'wl=520','-fx'], wait=1.0)
        os.remove('temp.png')

        # Get the image from the camera
        image = None
        while image is None:
            image = camera.get_image(pil_image = False, eight_bit = False)

        im.set_data(image)
        ax.set_title('Frame {}'.format(i))
        plt.pause(1)

        if i > 3:
            break
    

    camera.clean_up()
