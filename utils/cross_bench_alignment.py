#
import os
import yaml
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu

from diffractive_optical_model.diffractive_optical_model import DOM

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


def create_cross(nx, ny, offset_x, offset_y, cross_size=100):
    # Return a cross image with center at the offset. Min value is 0, max value is 255
    # Image size is nx x ny
    cross = np.zeros((nx, ny), dtype=np.uint8)
    cross_x = nx // 2 + offset_x
    cross_y = ny // 2 + offset_y
    cross[cross_x - cross_size//2:cross_x + cross_size//2, :] = 255
    cross[:, cross_y - cross_size//2:cross_y + cross_size//2] = 255

    return cross


if __name__ == "__main__":

    import sys
    sys.path.append('../')
    import time

    # Read the CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image to upload to the SLM')
    parser.add_argument('--f' , type=float, default=0, help='Focal length of the lens')
    parser.add_argument('--wl', type=float, default=0, help='Wavelength of the light')
    parser.add_argument('--x', type=int, default=0, help='x offset of the image')
    parser.add_argument('--y', type=int, default=0, help='y offset of the image')
    parser.add_argument('--theta', type=float, default=0, help='Rotation of the image')
    parser.add_argument("--shf", type=int, default=4, help="Image manipulation flag")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between updates")
    parser.add_argument('--ip', type=str, default = '10.10.80.1', help='IP address of the SLM to upload to')
    args = parser.parse_args()

    # Load the config file
    config = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)

    # Initialize the SLM
    slm0 = holoeye_pluto21.HoloeyePluto(host=args.ip)

    # Create the slm options
    options = ['shf=1', 'wl=520','-fx', '-q']

    # Initialize the camera
    camera = thorlabs_cc215mu.Thorlabs_CC215MU()

    # Get the intial plane params from the config file
    plane_params = config['planes'][0]

    # Get the spatial extent of the plane
    lx, ly = plane_params['size']
    nx, ny = plane_params['Nx'], plane_params['Ny']

    # Generate an intial image to send to the SLM
    image = np.zeros((nx, ny), dtype=np.uint8)

    # Upload the initial image to the SLM
    send_to_slm(slm0, image)
    update_slm(slm0, options)
    time.sleep(1)

    # Get a background image
    image = None
    while image is None:
        image = camera.get_image(pil_image = False, eight_bit = False)

    background_image = np.asarray(image)

    # Initialize the DOM
    dom = DOM(config)

    fig, ax = plt.subplots(4,1)
    im0 = ax[0].imshow(background_image, cmap='gray', vmin=0, vmax=255)
    im1 = ax[1].imshow(background_image, cmap='gray', vmin=0, vmax=1)
    im2 = ax[2].imshow(background_image, cmap='gray', vmin=0, vmax=1)
    im3 = ax[3].imshow(background_image, cmap='gray', vmin=0, vmax=1)

    cross = create_cross(nx, ny, args.x, args.y, cross_size=50)
    send_to_slm(slm0, cross)
    update_slm(slm0, options)
    time.sleep(1)

    cross_for_dom = torch.tensor(cross, dtype=torch.complex64).unsqueeze(0)
    output = dom.forward(cross_for_dom)
    output = output.squeeze().abs().cpu().numpy()**2
    output = output / np.max(output)

    counter = 0
    while(1):
        image = None
        while image is None:
            image = camera.get_image(pil_image = False, eight_bit = False)

        camera_image = image - background_image
        camera_image = camera_image / np.max(camera_image)
        difference = np.abs(camera_image - output)
        im0.set_data(cross)
        im1.set_data(camera_image)
        im2.set_data(output)
        im3.set_data(difference)

        fig.suptitle('Frame {}'.format(counter))
        plt.pause(args.delay)

        counter += 1

