#
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import pickle

sys.path.append('../')
sys.path.append('../../')


from optics_benchtop import thorlabs_cc215mu


if __name__ == "__main__":

    camera = thorlabs_cc215mu.Thorlabs_CC215MU()
    min_us, max_us = camera.camera.exposure_time_range_us
    print("Min exposure time: {} us".format(min_us))
    print("Max exposure time: {} us".format(max_us))

    exposure_times = np.linspace(min_us, max_us, 100)
    camera.clean_up()

    times = {}
    for et in tqdm(exposure_times):
        et = int(et)
        camera = thorlabs_cc215mu.Thorlabs_CC215MU(exposure_time_us = et)
        time.sleep(1)
        times[et] = []
        
        for i in tqdm(range(0,10), leave=False, desc="Exposure time: {} us".format(et)):
            start = time.time()
            image = None
            while image is None:
                image = camera.get_image(pil_image = False, eight_bit = True)
            end = time.time()
            times[et].append(end-start)
        times[et] = np.asarray(times[et])

        camera.clean_up()

    pickle.dump(times, open('exposure_times.pkl', 'wb'))
