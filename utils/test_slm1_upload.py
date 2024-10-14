from PIL import Image
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

import sys
sys.path.append('../..')

from optics_benchtop import holoeye_pluto21
from diffractive_optical_model.diffractive_optical_model import DOM


def send_to_slm(slm, image):
    img = Image.fromarray(image)
    img.save("temp.png")
    slm.send_scp("temp.png")

def update_slm(slm, options=None):
    if options is None:
        options = ['wl=520', '-q']
    slm.update(filename = '/root/temp.png', options=options, wait=None)
    os.remove("temp.png")


if __name__ == "__main__":

    config = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader)
    ip = "10.10.81.1"
    slm = holoeye_pluto21.HoloeyePluto(host=ip)

    dom = DOM(config)

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

    while(1):
        send_to_slm(slm, lens_phase)
        update_slm(slm)
        print("Updated SLM")
        time.sleep(2)
