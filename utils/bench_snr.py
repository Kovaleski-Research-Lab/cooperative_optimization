#
import os
import sys
import time
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu
from src.datamodule.datamodule import select_data

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
    sys.path.append('../')
    sys.path.append('../../')
    from src.utils.spatial_resample import spatial_resample
    sys.path.append('../../diffractive_optical_model/diffractive_optical_model/plane')
    from plane import Plane

    # Load the config file
    config = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)
    config['paths']['path_root'] = '../'

    # Initialize the SLM
    slm0 = holoeye_pluto21.HoloeyePluto(host='10.10.80.1')

    # Load the data module
    datamodule = select_data(config)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    # Get 10 images from the dataloader
    images = []
    for i in range(10):
        batch = next(iter(train_loader))
        images.append(batch[1])

    # Create the slm options
    options = ['shf=1', 'wl=520','-fx', '-q']

    # Get the intial plane params from the config file
    plane_params = config['planes'][0]

    # Get the spatial extent of the plane
    lx, ly = plane_params['size']
    nx, ny = plane_params['Nx'], plane_params['Ny']

    # Zeros background object for slm
    dark_object = np.zeros((nx, ny), dtype=np.uint8)

    # 255 background object for slm
    bright_object = np.ones((nx, ny), dtype=np.uint8) * 255

    # Create the plane
    plane = Plane(plane_params)
    scaled_plane = plane.scale(0.53, inplace=False)
    output_plane_params = config['planes'][2]
    output_plane = Plane(output_plane_params)

    # Resample the images to the SLM size
    resampled_images = []
    for image in images:
        resampled_images.append(spatial_resample(scaled_plane, image, output_plane))

    pickle.dump(resampled_images, open('resampled_images.pkl', 'wb'))


    # --------------------------------
    # Process for comparing SNR
    # --------------------------------
    # For each exposure time:
    #  Create a camera with the exposure time
    #  Send the dark background image to the SLM
    #  Collect the dark background image
    #  Send the bright background image to the SLM
    #  Collect the bright background image
    #  For each image:
    #   Send the image to the SLM
    #   Collect the image from the camera
    #  Save the dark background image
    #  Save the bright background image
    #  Save the collected images

    # Initialize the range of exposure times
    exposure_times_us = np.linspace(100, 500000, 100)


    for et in tqdm(exposure_times_us):
        # Initialize the dictionary to store the data
        data = {}

        et = int(et)
        # Initialize the camera
        camera = thorlabs_cc215mu.Thorlabs_CC215MU(exposure_time_us = et)

        # Send the dark background image to the SLM
        send_to_slm(slm0, dark_object)
        update_slm(slm0, options)
        time.sleep(1)

        # Collect the dark background image
        image = None
        camera.camera.issue_software_trigger()
        while image is None:
            image = camera.get_image(pil_image = False, eight_bit = False)

        dark_background_image = np.asarray(image)

        # Send the bright background image to the SLM
        send_to_slm(slm0, bright_object)
        update_slm(slm0, options)
        time.sleep(1)

        # Collect the bright background image
        image = None
        camera.camera.issue_software_trigger()
        while image is None:
            image = camera.get_image(pil_image = False, eight_bit = False)

        bright_background_image = np.asarray(image)

        # Initialize the list to store the collected images
        collected_images = []

        for image in tqdm(images, desc="Collecting images", leave=False):
            # Send the image to the SLM
            send_to_slm(slm0, image.squeeze().numpy())
            update_slm(slm0, options)
            time.sleep(1)

            # Collect the image from the camera
            image = None
            camera.camera.issue_software_trigger()
            while image is None:
                image = camera.get_image(pil_image = False, eight_bit = False)

            collected_images.append(np.asarray(image))

        # Save the dark background image
        data['dark_background'] = dark_background_image
        # Save the bright background image
        data['bright_background'] = bright_background_image
        # Save the collected images
        data['collected_images'] = collected_images

        camera.clean_up()

        # Save the data
        pickle.dump(data, open(f'snr_data_{et:010d}.pkl', 'wb'))
