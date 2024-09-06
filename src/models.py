import os
import sys
from loguru import logger
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl


sys.path.append('../')
sys.path.append('../../')

from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu
from diffractive_optical_model import don


class DigitalModel(pl.LightningModule):
    def __init__(self, params):
        self.params = params

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['lr'])

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

class CooperativeOpticalModel(don.DON):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.alpha, self.beta, self.gamma = params['alpha'], params['beta'], params['gamma']
        self.slm0 = holoeye_pluto21.HoloeyePluto(host=params['slm0_host'])
        self.camera = thorlabs_cc215mu.Thorlabs_CC215MU()

        if params['display']:
            self.init_display()
            self.path_images = params['paths']['path_images']
            os.makedirs(self.path_images, exist_ok=True)

        #self.double()

    def init_display(self):
        self.fig, self.ax = plt.subplot_mosaic('abc;def;ghi', figsize=(20, 10))
        # Direct outputs and the target
        self.im0 = self.ax['a'].imshow(np.zeros((1080, 1920)), cmap='gray')
        self.im1 = self.ax['b'].imshow(np.zeros((1080, 1920)), cmap='gray')
        self.im2 = self.ax['c'].imshow(np.zeros((1080, 1920)), cmap='gray')

        # The differences
        self.im3 = self.ax['d'].imshow(np.zeros((1080, 1920)), cmap='gray')
        self.im4 = self.ax['e'].imshow(np.zeros((1080, 1920)), cmap='gray')
        self.im5 = self.ax['f'].imshow(np.zeros((1080, 1920)), cmap='gray')

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

        simulation_intensity = simulation_outputs['intensity'].cpu().detach().numpy().squeeze()
        image = image.cpu().numpy().squeeze()
        target = target.cpu().abs().numpy().squeeze()
        phase = torch.flip(phase.squeeze(), (0,1)).cpu().detach().numpy()

        self.im0.set_data(simulation_intensity)
        self.im0.autoscale()
        self.im1.set_data(image)
        self.im1.autoscale()
        self.im2.set_data(target)
        self.im2.autoscale()

        self.im3.set_data(simulation_intensity - image)
        self.im3.autoscale()
        self.im4.set_data(simulation_intensity - target)
        self.im4.autoscale()
        self.im5.set_data(image - target)
        self.im5.autoscale()

        self.im6.set_data(phase)
        self.im6.autoscale()

        self.fig.suptitle("Epoch {}".format(self.current_epoch), fontsize=16)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

        self.fig.savefig(os.path.join(self.path_images, 'epoch_{:03d}.png'.format(self.current_epoch)))

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

        simulation_output =simulation_output.squeeze()
        collected_image = collected_image.squeeze()
        target = target.abs().double().squeeze()

        # Simulation to target
        sim_to_target = torch.nn.functional.mse_loss(simulation_output, target)

        # Simulation to collected image
        sim_to_collected = torch.nn.functional.mse_loss(simulation_output, collected_image)

        # Collected image to target
        # Here, we need to set the data of the simulation to the data of the collected image
        # to use the gradient of the simulation to do backpropagation
        # Need to copy it to avoid messing up the original simulation output
        simulation_output_copy = simulation_output.clone()
        simulation_output_copy.data = collected_image
        collected_to_target = torch.nn.functional.mse_loss(simulation_output_copy, target)

        # Total loss
        loss = self.alpha * sim_to_target + self.beta * sim_to_collected + self.gamma * collected_to_target

        return loss, sim_to_target, sim_to_collected, collected_to_target

    def sim_forward(self, u:torch.Tensor):
        # Iterate through the layers
        for i, layer in enumerate(self.layers):
            u = layer(u)
        return u

    def bench_forward(self):
        # Get the phases from the simualtion modulator
        phases = self.layers[0].modulator.get_phase(with_grad = False)

        # Phase wrap - [0, 2pi]
        phases = phases % (2 * torch.pi)

        # Convert to [0, 1]
        phases = phases / (2 * torch.pi)

        # Invert
        phases = torch.abs(1-phases)

        # Scale to [0, 255]
        phases = phases * 255

        # Convert to numpy uint8
        slm_sample = phases.cpu().numpy().squeeze().astype(np.uint8)

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

        image = torch.from_numpy(image).to(self.device) * 10

        return image, phases

    def get_auxilary_outputs(self, wavefront):
        amplitude = torch.abs(wavefront)/10
        normalized_amplitude = amplitude / torch.max(amplitude)
        phase = torch.angle(wavefront)
        intensity = amplitude ** 2
        normalized_intensity = intensity / torch.max(intensity)

        return {'amplitude': amplitude.double(),
                'normalized_amplitude': normalized_amplitude.double(),
                'phase': phase.double(),
                'intensity': intensity.double(),
                'normalized_intensity': normalized_intensity.double()}

    def shared_step(self, batch):

        # Run the simulation forward
        wavefront = self.sim_forward(batch[0])

        # Get the image from the camera
        image, phases = self.bench_forward()

        # Flip the image
        #image = torch.flip(image, (0,1))

        # Convert to double
        image = image.double()

        # From the simulated wavefronts, get the auxilary outputs
        simulation_outputs = self.get_auxilary_outputs(wavefront)

        # Update the display if needed
        if self.params['display']:
            self.update_display(simulation_outputs, image, batch[2], phases)

        return {'simulation_outputs': simulation_outputs,
                'image': image}

    def on_train_start(self):
        self.upload_benign_image()

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        # Get the loss
        loss, sim_to_target, sim_to_collected, collected_to_target = self.objective(outputs['simulation_outputs']['amplitude'], outputs['image'], batch[2]) 
        # Log the loss
        self.log('loss_train', loss, prog_bar=True)
        self.log('sim_to_target_train', sim_to_target)
        self.log('sim_to_collected_train', sim_to_collected)
        self.log('collected_to_target_train', collected_to_target)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        # Get the loss
        loss, sim_to_target, sim_to_collected, collected_to_target = self.objective(outputs['simulation_outputs']['amplitude'], outputs['image'], batch[2]) 
        # Log the loss
        self.log('loss_val', loss, prog_bar=True)
        self.log('sim_to_target_val', sim_to_target)
        self.log('sim_to_collected_val', sim_to_collected)
        self.log('collected_to_target_val', collected_to_target)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        # Get the loss
        loss, sim_to_target, sim_to_collected, collected_to_target = self.objective(outputs['simulation_outputs']['amplitude'], outputs['image'], batch[2])
        # Log the loss
        self.log('loss_test', loss, prog_bar=False)
        self.log('sim_to_target_test', sim_to_target)
        self.log('sim_to_collected_test', sim_to_collected)
        self.log('collected_to_target_test', collected_to_target)
        return loss


