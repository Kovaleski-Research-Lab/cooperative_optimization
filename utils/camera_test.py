

import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append('../')
sys.path.append('../../')


from optics_benchtop import thorlabs_cc215mu


if __name__ == "__main__":

    camera = thorlabs_cc215mu.Thorlabs_CC215MU(exposure_time_us = 100000)


    image = None

    while image is None:
        image = camera.get_image(pil_image = False, eight_bit = False)

    fig, ax = plt.subplots(1, 1, figsize=(8.85, 5))
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=1)

    counter = 0
    while True:
        while image is None:
            image = camera.get_image(pil_image = False, eight_bit = False)
        if image is not None:
            im.set_data(image)
            max_val = np.max(image)
            shape = image.shape
            ax.axis('off')
            ax.axvline(shape[1]//2, 0, shape[0], color='r')
            ax.axhline(shape[0]//2, 0, shape[1], color='r')
            ax.set_title('Frame {}, Max val {}'.format(counter, max_val))
            plt.pause(0.1)
            counter += 1


    camera.clean_up()

