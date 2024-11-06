#
import os
import yaml
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import torch

import requests
import sys
sys.path.append('../')
sys.path.append('../../')
from src.datamodule.datamodule import select_data
from src.utils.spatial_resample import spatial_resample
from diffractive_optical_model.diffractive_optical_model import DOM

def add_slm(slm_name, slm_host, bench_server_url, add_slm_endpoint):
    # Add an SLM
    slm_params = {  'slm_name': slm_name,
                    'slm_host': slm_host,
                  }
    response = requests.post(bench_server_url + add_slm_endpoint, json=slm_params)
    return response

def add_camera(camera_name, camera_exposure_time, bench_server_url, add_camera_endpoint):
    # Add a camera
    camera_params = {   'camera_name': camera_name,
                        'camera_exposure_time': camera_exposure_time,
                    }
    response = requests.post(bench_server_url + add_camera_endpoint, json=camera_params)
    return response

def get_camera_image(camera_name, bench_server_url, get_camera_image_endpoint):
    # Get an image from the camera
    payload = {'camera_name': camera_name}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(bench_server_url + get_camera_image_endpoint, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'ok':
            image = base64.b64decode(data['image'])
            image = BytesIO(image)
            image = np.load(image)
            return image
        else:
            print(f"Error : {data['message']}")
            return None
    else:
        print(f"Error : {response.status_code} - {response.reason}")
        return None

def send_to_slm(slm_name, options, wait, bench_server_url, update_slm_endpoint, image):
    # If the image is a numpy array, convert it to a PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    files = {'image': ('image.png', buffered, 'image/png')}
    payload = {'slm_name': slm_name, 'options': options, 'wait': wait}
    response = requests.post(bench_server_url + update_slm_endpoint, files=files, data=payload)
    return response


def get_bench_info(bench_server_url):
    # Get the bench info
    info_api_endpoint = '/info'
    response = requests.get(bench_server_url + info_api_endpoint)
    return response

def get_lens_from_dom(dom):
    # Get the phases from the simualtion modulator
    phases = dom.layers[1].modulator.get_phase(with_grad = False)

    # Phase wrap - [0, 2pi]
    phases = phases % (2 * torch.pi)

    # Convert to [0, 1]
    phases = phases / (2 * torch.pi)

    # Invert
    phases = torch.abs(1-phases)

    # Scale to [0, 255]
    phases = phases * 255

    # Convert to numpy uint8
    lens_phase = phases.cpu().numpy().squeeze().astype(np.uint8)

    return lens_phase

if __name__ == "__main__":

    import sys
    sys.path.append('../')
    import time

    # Read the CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('--wl', type=float, default=0, help='Wavelength of the light')
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between updates")
    args = parser.parse_args()

    # Load the config file
    config = yaml.load(open("../../config.yaml", "r"), Loader=yaml.FullLoader)
    config['paths']['path_root'] = '../../'

    # Bench server URL and endpoints
    bench_server_url = config['bench_server_url']
    add_slm_endpoint = config['add_slm_endpoint']
    remove_slm_endpoint = config['remove_slm_endpoint']
    add_camera_endpoint = config['add_camera_endpoint']
    remove_camera_endpoint = config['remove_camera_endpoint']
    update_slm_endpoint = config['update_slm_endpoint']
    get_camera_image_endpoint = config['get_camera_image_endpoint']
    reset_bench_endpoint = config['reset_bench_endpoint']

    # Load the data module
    datamodule = select_data(config)
    datamodule.setup()

    # Reset the bench
    response = requests.post(bench_server_url + reset_bench_endpoint)

    train_loader = datamodule.train_dataloader()

    # Create the slm options
    options0 = ['shf=1', 'wl=520','-fx', '-q']
    options1 = ['shf=1', 'wl=520','-fx', '-q']

    # Create SLMs
    slm0_params = config['slms'][0]
    slm0_name = slm0_params['name']
    slm0_host = slm0_params['host']

    slm1_params = config['slms'][1]
    slm1_name = slm1_params['name']
    slm1_host = slm1_params['host']
    response = add_slm(slm0_name, slm0_host, bench_server_url, add_slm_endpoint)
    response = add_slm(slm1_name, slm1_host, bench_server_url, add_slm_endpoint)

    # Create camera
    camera_name = config['camera_name']
    camera_exposure_time = config['camera_exposure_time']
    response = add_camera(camera_name, camera_exposure_time, bench_server_url, add_camera_endpoint)
    print(get_bench_info(bench_server_url).text)

    # Initialize a DOM simulation
    dom = DOM(config)

    scaled_plane = dom.layers[0].input_plane.scale(0.53, inplace=False)

    # Get the lens phases from the DOM
    lens_phase = get_lens_from_dom(dom)

    # Send the lens phase to slm1
    wait = 0
    response = send_to_slm(slm1_name, options1, wait, bench_server_url, update_slm_endpoint, lens_phase)

    # Get the intial plane params from the config file
    plane_params = config['planes'][0]

    # Get the spatial extent of the plane
    lx, ly = plane_params['size']
    nx, ny = plane_params['Nx'], plane_params['Ny']

    # Generate an intial image to send to the SLM
    image = np.zeros((nx, ny), dtype=np.uint8)

    # Send the image to the SLM0
    response = send_to_slm(slm0_name, options0, wait, bench_server_url, update_slm_endpoint, image)

    # Get a background image
    image = get_camera_image(camera_name, bench_server_url, get_camera_image_endpoint)
    background_image = np.asarray(image)

    fig, ax = plt.subplots(3,1)
    im0 = ax[0].imshow(background_image, cmap='gray', vmin=0, vmax=1)
    im1 = ax[1].imshow(background_image, cmap='gray', vmin=0, vmax=1)
    im2 = ax[2].imshow(background_image, cmap='gray', vmin=0, vmax=1)

    counter = 0
    while(1):
        batch = next(iter(train_loader))
        images, slm_sample, labels = batch
        # Send to slm
        response = send_to_slm(slm0_name, options0, 0.5, bench_server_url, update_slm_endpoint, slm_sample.squeeze().numpy())
        time.sleep(args.delay)

        images = spatial_resample(scaled_plane, images.abs(), dom.layers[1].output_plane).squeeze()

        # Get camera image
        image = get_camera_image(camera_name, bench_server_url, get_camera_image_endpoint)
        camera_image = image - background_image
        camera_image = camera_image / np.max(camera_image)
        im0.set_data(camera_image)
        im1.set_data(images.squeeze().abs().numpy())

        difference = np.abs(camera_image - images.squeeze().abs().numpy())
        difference = difference / np.max(difference)
        im2.set_data(difference)
        fig.suptitle('Frame {}'.format(counter))
        plt.pause(args.delay)
        counter += 1
