

import os
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":


    files = os.listdir('results_spt')
    files = [f for f in files if f.endswith('.npy')]
    files = sorted(files)

    fig, ax = plt.subplots(1,1)
    im = ax.imshow(np.load('results_spt/' + files[0]), cmap='gray', vmin=0, vmax=0.05)

    counter = 0
    for f in files:
        image = np.load('results_spt/' + f)
        im.set_array(image)
        ax.set_title("{}".format(counter))
        plt.pause(0.5)
        counter +=1
