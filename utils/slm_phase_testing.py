
# This file compares the png pixel values sent to the SLM with the overall
# intensity at the camera. This is intended for the first SLM in the stack
# which simulates the binary object. The goal is to map the image pixel values
# to the corresponding intesity values at the camera. We hope see a linear increase
# in intensity as the pixel values increase (0-255). Functinoally, increasing 
# pixel values should increase the overal phase delay of the wavefront, allowing more
# of the wavefront to pass through the polarizing beam splitter.

import os
import sys
import yaml
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu

def send_to_slm(slm, image):
    img = Image.fromarray(image)
    img.save('temp.png')
    slm.send_scp('temp.png')

def update_slm(slm, options=None):
    if options is None:
        options = ['wl=520', '-q']

    slm.update(filename='/root/temp.png', options=options, wait=None)
    os.remove('temp.png')

def upload_benign_image(slm):
    image = np.zeros((1080, 1920), dtype=np.uint8)
    img = Image.fromarray(image)
    img.save('benign.png')
    slm.send_scp('benign.png')
    slm.update(filename='/root/benign.png', options=['wl=520', '-fx'], wait=None)

if __name__ == "__main__":
    # Parse the CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = '../config.yaml', help='Path to the config file')
    parser.add_argument('--path_results', type=str, default = 'results_spt/with_cal', help='Path to the folder of results')
    parser.add_argument('--f' , type=float, default=0, help='Focal length of the lens')
    parser.add_argument('--wl', type=float, default=0, help='Wavelength of the light')
    parser.add_argument('--x', type=float, default=0, help='x offset of the image')
    parser.add_argument('--y', type=float, default=0, help='y offset of the image')
    parser.add_argument('--theta', type=float, default=0, help='Rotation of the image')
    parser.add_argument("--shf", type=int, default=4, help="Image manipulation flag")
    args = parser.parse_args()

    path_config = args.config
    path_results = args.path_results

    os.makedirs(path_results, exist_ok=True)

    # Load the config file
    config = yaml.load(open(path_config, "r"), Loader=yaml.FullLoader)

    # Set the options for the SLM
    options = ['wl={}'.format(args.wl), 'x={}'.format(args.x), 'y={}'.format(args.y), 'f={}'.format(args.f), '-q', '-fx', 'shf={}'.format(args.shf)]

    # Initialize the SLM
    slm0 = holoeye_pluto21.HoloeyePluto(host=config['slm0_host'])

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

    # Initialize a list to store the intensity values
    intensity_list = []

    # Loop through increasing pixel values (0-255)
    for i in tqdm(range(0, 256)):
        # Create the image
        sample = np.ones((nx, ny), dtype=np.uint8) * i

        # Send the image to the SLM
        send_to_slm(slm0, sample)

        # Update the SLM
        update_slm(slm0, options)

        # Get 10 images from the camera
        intensities = []
        for j in range(10):
            image = None
            while image is None:
                image = camera.get_image(pil_image = False, eight_bit = False)
            image = np.asarray(image)
            # Make sure the camera is not saturated
            if ((image == 1).sum() > 10):
               upload_benign_image(slm0)
               print('Camera is saturated')
               exit()
            intensities.append(image.mean()) 

        # Calculate the average intensity of the images at the camera
        intensity = np.mean(intensities) 

        # Save the image from the camera to a file
        np.save(os.path.join(path_results, 'image_{0:04d}.npy'.format(i)), image)

        # Append the intensity to a list
        intensity_list.append(intensity)

        # Pause for 10ms
        time.sleep(0.10)

    # Upload a benign image to the SLM
    upload_benign_image(slm0)

    # Save the intensity list to a file
    np.savetxt(os.path.join(path_results,'intensity_list.txt'), intensity_list)

    # Plot the intensity list as a fucntion of pixel value
    plt.plot(intensity_list)
    plt.xlabel('Pixel Value')
    plt.ylabel('Intensity')
    plt.show()

    # Clean up
    camera.clean_up()

