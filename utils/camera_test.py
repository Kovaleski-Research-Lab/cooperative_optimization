

import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append('../')
sys.path.append('../../')


from optics_benchtop import thorlabs_cc215mu


if __name__ == "__main__":

    camera = thorlabs_cc215mu.Thorlabs_CC215MU()

    image = None

    while image is None:
        image = camera.get_image(pil_image = False, eight_bit = False)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=1)

    counter = 0
    while True:
        image = camera.get_image(pil_image = False, eight_bit = False)
        if image is not None:
            im.set_data(image)
            ax.set_title('Frame {}'.format(counter))
            plt.pause(0.1)
            counter += 1
            if counter > 10:
                break

    camera.clean_up()

