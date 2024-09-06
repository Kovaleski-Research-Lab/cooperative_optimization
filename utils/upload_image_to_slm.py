#
import os
import yaml
import argparse
import numpy as np
from PIL import Image

import sys
sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21


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

    # Read the CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image to upload to the SLM')
    parser.add_argument('--f' , type=float, default=0, help='Focal length of the lens')
    parser.add_argument('--wl', type=float, default=0, help='Wavelength of the light')
    parser.add_argument('--x', type=float, default=0, help='x offset of the image')
    parser.add_argument('--y', type=float, default=0, help='y offset of the image')
    parser.add_argument('--theta', type=float, default=0, help='Rotation of the image')
    parser.add_argument("--shf", type=int, default=4, help="Image manipulation flag")

    args = parser.parse_args()

    # Load the config file
    config = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)

    # Initialize the SLM
    slm0 = holoeye_pluto21.HoloeyePluto(host=config['slm0_host'])

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

    # Update the SLM with the image
    image = Image.open(args.image)
    image = image.rotate(angle = args.theta, resample=Image.BICUBIC)

    # Send the image to the SLM
    send_to_slm(slm0, image)

    # Update the SLM with the image
    options = ['wl={}'.format(args.wl), 'x={}'.format(args.x), 'y={}'.format(args.y), 'f={}'.format(args.f), '-q', '-fx', 'shf={}'.format(args.shf)]
    update_slm(slm0, options=options)

