import os
import yaml
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21

from optics_benchtop import thorlabs_cc215mu

def send_to_slm(slm, image):
    # If the image is a numpy array, convert it to a PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image.save('temp.png')
    slm.send_scp('temp.png')

def update_slm(slm, options=None):
    if options is None:
        options = ['wl=520', '-q']

    slm.update(filename='/root/temp.png', options=options, wait=None)
    os.remove('temp.png')

if __name__ == "__main__":

    import sys
    sys.path.append('../')
    from src.datamodule import select_data
    import time

    # Read the CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image to upload to the SLM')
    parser.add_argument('--f' , type=float, default=0, help='Focal length of the lens')
    parser.add_argument('--wl', type=float, default=0, help='Wavelength of the light')
    parser.add_argument('--x', type=float, default=0, help='x offset of the image')
    parser.add_argument('--y', type=float, default=0, help='y offset of the image')
    parser.add_argument('--theta', type=float, default=0, help='Rotation of the image')
    parser.add_argument("--shf", type=int, default=4, help="Image manipulation flag")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between updates")
    parser.add_argument('--ip', type=str, default = '10.10.80.1', help='IP address of the SLM to upload to')
    args = parser.parse_args()

    # Load the config file
    config = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)

    # Initialize the SLM
    slm0 = holoeye_pluto21.HoloeyePluto(host=args.ip)

    # Get the intial plane params from the config file
    plane_params = config['planes'][0]

    # Get the spatial extent of the plane
    lx, ly = plane_params['size']
    nx, ny = plane_params['Nx'], plane_params['Ny']

    # Generate an intial image to send to the SLM
    image = np.zeros((nx, ny), dtype=np.uint8)

    # Upload the initial image to the SLM
    send_to_slm(slm0, image)
    update_slm(slm0)

    config['batch_size'] = 1
    config['model_id'] = "test_0"
    config['paths']['path_root'] = '../'

    dm = select_data(config)
    dm.prepare_data()
    dm.setup(stage="fit")

    #View some of the data
    images,slm_sample, labels = next(iter(dm.train_dataloader()))

    #camera = thorlabs_cc215mu.Thorlabs_CC215MU()

    for batch in dm.train_dataloader():
        images,slm_sample, labels = batch
        send_to_slm(slm0, slm_sample.squeeze().numpy())
        # Update the SLM with the image
        options = ['wl={}'.format(args.wl), 'x={}'.format(args.x), 'y={}'.format(args.y), 'f={}'.format(args.f), '-q', '-fx', 'shf={}'.format(args.shf)]
        update_slm(slm0, options=options)

        time.sleep(args.delay)
        #image = None
        #while image is None:
        #    image = camera.get_image(pil_image = False, eight_bit = False)

        #image = np.asarray(image)
        #fig, ax = plt.subplots(1, 1)
        #im = ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        #fig.savefig('image_{}.png'.format(time.time()))
        #plt.close('all')



