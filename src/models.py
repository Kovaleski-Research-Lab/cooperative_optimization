import os
import sys
from loguru import logger
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import torch


sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu
from diffractive_optical_model import don


class CooperativeOpticalModel(don.DON):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.alpha, self.beta, self.gamma = params['alpha'], params['beta'], params['gamma']
        self.slm0 = holoeye_pluto21.HoloeyePluto(host=params['slm0_host'])
        self.camera = thorlabs_cc215mu.Thorlabs_CC215MU()

        if params['display']:
            self.init_display()

    def init_display(self):
        self.fig, self.ax = plt.subplot_mosaic('abc;def;ghi', figsize=(20, 10))
        # Direct outputs and the target
        self.im0 = self.ax['a'].imshow(np.zeros((1080, 1920)), cmap='gray', vmin=0, vmax=1)
        self.im1 = self.ax['b'].imshow(np.zeros((1080, 1920)), cmap='gray', vmin=0, vmax=1)
        self.im2 = self.ax['c'].imshow(np.zeros((1080, 1920)), cmap='gray', vmin=0, vmax=1)

        # The differences
        self.im3 = self.ax['d'].imshow(np.zeros((1080, 1920)), cmap='gray', vmin=-1, vmax=1)
        self.im4 = self.ax['e'].imshow(np.zeros((1080, 1920)), cmap='gray', vmin=-1, vmax=1)
        self.im5 = self.ax['f'].imshow(np.zeros((1080, 1920)), cmap='gray', vmin=-1, vmax=1)

        # The phases
        self.im6 = self.ax['g'].imshow(np.zeros((1080, 1920)), cmap='gray', vmin=-np.pi, vmax=np.pi)

        for a in self.ax:
            self.ax[a].grid(True)

        self.ax['h'].axis('off')
        self.ax['h'].grid(False)
        self.ax['i'].axis('off')
        self.ax['i'].grid(False)

        self.ax['a'].set_title("Simulated Intensity")
        self.ax['b'].set_title("Camera Image")
        self.ax['c'].set_title("Target Image")

        self.ax['d'].set_title("Simulated - Camera")
        self.ax['e'].set_title("Simulated - Target")
        self.ax['f'].set_title("Camera - Target")

        self.ax['g'].set_title("Phase")

        self.fig.suptitle("Epoch 0", fontsize=16)
        plt.tight_layout()
        plt.ion()
        self.fig.show()

    def update_display(self, simulation_outputs, image, target, phase):

        self.im0.set_data(simulation_outputs['intensity'].cpu().numpy())
        self.im1.set_data(image.cpu().numpy())
        self.im2.set_data(target.cpu().numpy())

        self.im3.set_data(simulation_outputs['intensity'].cpu().numpy() - image.cpu().numpy())
        self.im4.set_data(simulation_outputs['intensity'].cpu().numpy() - target.cpu().numpy())
        self.im5.set_data(image.cpu().numpy() - target.cpu().numpy())

        self.im6.set_data(phase.cpu().numpy())

        self.fig.suptitle("Epoch {}".format(self.current_epoch), fontsize=16)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def send_to_slm(self, slm_sample):
        logger.info("Sending image to SLM")
        img = Image.fromarray(slm_sample)
        img.save('temp.png')
        self.slm0.send_scp('temp.png')

    def update_slm(self):
        logger.info("Updating SLM")
        self.slm0.update(filename='/root/temp.png', options=['wl=520', '-fx'], wait=None)
        os.remove('temp.png')

    def upload_benign_image(self):
        logger.warning("Uploading benign image to SLM")
        image = np.zeros((1080, 1920), dtype=np.uint8)
        img = Image.fromarray(image)
        img.save('benign.png')
        self.slm0.send_scp('benign.png')
        self.slm0.update(filename='/root/benign.png', options=['wl=520', '-fx'], wait=None)

    def objective(self, simulation_output, collected_image, target):

        # Simulation to target
        sim_to_target = torch.nn.functional.mse_loss(simulation_output, target)

        # Simulation to collected image
        sim_to_collected = torch.nn.functional.mse_loss(simulation_output, collected_image)

        # Collected image to target
        collected_to_target = torch.nn.functional.mse_loss(collected_image, target)

        # Total loss
        loss = self.alpha * sim_to_target + self.beta * sim_to_collected + self.gamma * collected_to_target


    def sim_forward(self, u:torch.Tensor):
        # Iterate through the layers
        for i, layer in enumerate(self.layers):
            u = layer(u)
        return u

    def bench_forward(self):
        # Get the phases from the simualtion modulator
        phases = self.layers[0].modulator.get_phases(with_grad = False)

        # Phase wrap - [0, 2pi]
        phases = phases % (2 * torch.pi)

        # Convert to [0, 1]
        phases = phases / (2 * torch.pi)

        # Invert
        phases = torch.abs(1-phases)

        # Scale to [0, 255]
        phases = phases * 255

        # Convert to numpy uint8
        slm_sample = phases.cpu().numpy().astype(np.uint8)

        # Send to SLM
        self.send_to_slm(slm_sample)

        # Update the SLM
        self.update_slm()

        # Get the image from the camera
        image = None
        while image is None:
            image = self.camera.get_image(pil_image = False, eight_bit = False)

        # Make sure the camera is not saturated
        if ((image == 1).sum() > 10):
            logger.error("Camera is saturated")
            self.upload_benign_image()

        return image, phases

    def get_auxilary_outputs(self, wavefront):
        amplitude = torch.abs(wavefront)
        normalized_amplitude = amplitude / torch.max(amplitude)
        phase = torch.angle(wavefront)
        intensity = amplitude ** 2
        normalized_intensity = intensity / torch.max(intensity)

        return {'amplitude': amplitude,
                'normalized_amplitude': normalized_amplitude,
                'phase': phase,
                'intensity': intensity,
                'normalized_intensity': normalized_intensity}

    def shared_step(self, batch):

        # Run the simulation forward
        wavefront = self.sim_forward(batch[0])

        # Get the image from the camera
        image, phases = self.bench_forward()

        # Convert to torch tensor
        image = torch.tensor(image)

        # Flip the image
        image = torch.flip(image, (0,1))

        # Convert to double
        image = image.double()

        # From the simulated wavefronts, get the auxilary outputs
        simulation_outputs = self.get_auxilary_outputs(wavefront)

        # Update the display if needed
        if self.params['display']:
            self.update_display(simulation_outputs, image, batch[2], phases)

        return {'simulation_outputs': simulation_outputs,
                'image': image}

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        # Get the loss
        loss = self.objective(outputs['simulation_outputs']['intensity'], outputs['image'], batch[2])
        # Log the loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        # Get the loss
        loss = self.objective(outputs['simulation_outputs']['intensity'], outputs['image'], batch[2])
        # Log the loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        # Get the loss
        loss = self.objective(outputs['simulation_outputs']['intensity'], outputs['image'], batch[2])
        # Log the loss
        self.log('test_loss', loss)
        return loss


