import os
import sys
from loguru import logger
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from torchmetrics import F1Score, Accuracy, Precision, Recall, ConfusionMatrix
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


sys.path.append('/home/mblgh6/Documents/research/optics_benchtop')
sys.path.append('/home/mblgh6/Documents/research/cooperative_optimization')
from optics_benchtop import holoeye_pluto21
from optics_benchtop import thorlabs_cc215mu
from diffractive_optical_model.diffractive_optical_model import DOM
from src.utils.spatial_resample import spatial_resample

#-----------------------------------------
# Initialize: Classifier
#-----------------------------------------

class Classifier(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.params = params['classifier']
        self.paths = params['paths']
        self.num_classes = self.params['num_classes']
        self.architecture = self.params['architecture']
        self.learning_rate = self.params['learning_rate']
        self.transfer_learn = self.params['transfer_learn']
        self.freeze_backbone = self.params['freeze_backbone']

        self.select_model()

        self.f1 = F1Score(task = 'multiclass', num_classes=self.num_classes)
        self.accuracy = Accuracy(task = 'multiclass', num_classes=self.num_classes)
        self.precision = Precision(task = 'multiclass', num_classes=self.num_classes)
        self.recall = Recall(task = 'multiclass', num_classes=self.num_classes)
        self.cfm = ConfusionMatrix(task = 'multiclass', num_classes=self.num_classes)

        self.save_hyperparameters()

    def select_model(self):
        if self.transfer_learn:
            logger.info("Transfer learning")
        else:
            logger.info("Training from scratch")
        if self.architecture == 'resnet18':
            if self.transfer_learn:
                backbone = resnet18(weights = ResNet18_Weights.DEFAULT)
            else:
                backbone = resnet18(pretrained = False)
        elif self.architecture == 'resnet34':
            if self.transfer_learn:
                backbone = resnet34(weights = ResNet34_Weights.DEFAULT)
            else:
                backbone = resnet34(pretrained = False)
        elif self.architecture == 'resnet50':
            if self.transfer_learn:
                backbone = resnet50(weights = ResNet50_Weights.DEFAULT)
            else:
                backbone = resnet50(pretrained = False)
        else:
            raise ValueError("Architecture not supported")

        if self.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(num_filters, self.num_classes)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def calculate_metrics(self):
        pass

    def objective(self, outputs, targets):
        targets = torch.argmax(targets, dim=-1)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss

    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)
        return self.classifier(features)

    def shared_step(self, batch):
        sample, target = batch
        sample = torch.cat([sample, sample, sample], dim=1)
        prediction = self.forward(sample)
        return prediction, target

    def training_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch)
        loss = self.objective(outputs, targets)
        self.log('loss_train', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch)
        loss = self.objective(outputs, targets)
        self.log('loss_val', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch)
        loss = self.objective(outputs, targets)
        self.log('loss_test', loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss


#-----------------------------------------
# Initialize: Cooperative Optical Model
#-----------------------------------------

class CooperativeOpticalModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.paths = params['paths']
        self.classifier_image = params['classifier_image']
        self.alpha, self.beta, self.gamma, self.delta = params['alpha'], params['beta'], params['gamma'], params['delta']
        self.slm0 = holoeye_pluto21.HoloeyePluto(host=params['slm0_host'])
        self.slm1 = holoeye_pluto21.HoloeyePluto(host=params['slm1_host'])
        self.camera_exposure_time = int(params['exposure_time'])
        self.camera = thorlabs_cc215mu.Thorlabs_CC215MU(exposure_time_us=self.camera_exposure_time)
        self.dom = DOM(params)
        self.scaled_plane = self.dom.layers[0].input_plane.scale(0.53, inplace=False)
        self.classifier = Classifier(params).double()
        self.register_buffer('background_image', self.get_background_image())
    #-----------------------------------------
    # Initialize: Optimizer
    #-----------------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])

    #-----------------------------------------
    # Initialize: SLM utilities
    #-----------------------------------------
    def send_to_slm(self, slm_pattern, which = None):
        img = Image.fromarray(slm_pattern)
        if which == 0:
            img.save('temp0.png')
            logger.info("Sending image to SLM0")
            self.slm0.send_scp('temp0.png')
        elif which == 1:
            img.save('temp1.png')
            logger.info("Sending image to SLM1")
            self.slm1.send_scp('temp1.png')
        else:
            raise ValueError(f"SLM {which} not supported")

    def update_slm(self, which = None, wait=None):
        if which == 0:
            logger.info("Updating SLM0")
            self.slm0.update(filename='/root/temp0.png', options=['wl=520', '-fx', '-q'], wait=wait)
            os.remove('temp0.png')
        elif which == 1:
            logger.info("Updating SLM1")
            self.slm1.update(filename='/root/temp1.png', options=['wl=520', '-fx', '-q'], wait=wait)
            os.remove('temp1.png')
        else:
            raise ValueError(f"SLM {which} not supported")

    def upload_benign_image(self, which=None):
        logger.warning("Uploading benign image to SLM")
        image = np.zeros((1080, 1920), dtype=np.uint8)
        img = Image.fromarray(image)
        if which == 0:
            img.save('benign0.png')
            self.slm0.send_scp('benign0.png')
            self.slm0.update(filename='/root/benign0.png', options=['wl=520', '-fx', '-q'])
        elif which == 1:
            img.save('benign1.png')
            self.slm1.send_scp('benign1.png')
            self.slm1.update(filename='/root/benign1.png', options=['wl=520', '-fx', '-q'])
        else:
            raise ValueError(f"SLM {which} not supported")

    #-----------------------------------------
    # Initialize: Camera utilities
    #-----------------------------------------

    def get_background_image(self):
        self.upload_benign_image(which=0)
        return self.get_bench_image().double()

    def get_bench_image(self):
        # Get the image from the camera
        image = None
        self.camera.camera.issue_software_trigger()
        while image is None:
            image = self.camera.get_image(pil_image = False, eight_bit = False)
        image = torch.from_numpy(image).to(self.device)

        return image

    #-----------------------------------------
    # Initialize: Optical model utilities
    #-----------------------------------------

    def get_lens_from_dom(self):
        # Get the phases from the simualtion modulator
        phases = self.dom.layers[1].modulator.get_phase(with_grad = False)

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

    #-----------------------------------------
    # Initialize: Objective function
    #-----------------------------------------
    
    def objective(self, outputs, batch):
        # Parse the outputs
        simulation_outputs = outputs['simulation_outputs']
        simulation_images = simulation_outputs['images']
        bench_image = outputs['bench_image']
        classifier_output = outputs['classifier_output']
        classifier_target = outputs['classifier_target']

        # Parse the batch
        sample, slm_sample, target = batch

        # We are going to calculate all of the potential losses and select which to use
        # for the training using MCL parameters

        #-----------------------------------------
        # Image comparisons
        #-----------------------------------------
        # Need to resample the sample to the simulation output plane for comparison
        sample = spatial_resample(self.scaled_plane, sample.abs(), self.dom.layers[1].output_plane).squeeze()
        sim_to_ideal = torch.nn.functional.mse_loss(simulation_images, sample)
        sim_to_bench = torch.nn.functional.mse_loss(simulation_images, bench_image)

        # Bench to ideal requires copying the simulation image tensor to avoid messing up the original
        # and to allow for backpropagation using the simulation gradient
        simulation_image_copy = simulation_images.clone()
        simulation_image_copy.data = bench_image
        bench_to_ideal = torch.nn.functional.mse_loss(simulation_image_copy, sample)

        #-----------------------------------------
        # Classifier loss
        #-----------------------------------------
        classifier_loss = self.classifier.objective(classifier_output, classifier_target)


        # Total loss - right now a soft constrained MCL
        loss = self.alpha * sim_to_ideal + self.beta * sim_to_bench + self.gamma * bench_to_ideal + self.delta * classifier_loss

        # Return the loss and all the components for logging
        return loss, sim_to_ideal, sim_to_bench, bench_to_ideal, classifier_loss

    #-----------------------------------------
    # Initialize: Forward passes
    #-----------------------------------------

    def dom_forward(self, u:torch.Tensor):
        simulation_wavefront = self.dom.forward(u)
        simulation_outputs = self.dom.calculate_auxiliary_outputs(simulation_wavefront)
        return simulation_outputs

    def bench_forward(self, slm_sample):
        lens_phase = self.get_lens_from_dom()
        # Send to SLM
        slm_sample = slm_sample.squeeze().cpu().numpy().astype(np.uint8)
        self.send_to_slm(slm_sample, which=0)
        self.update_slm(which=0)
        self.send_to_slm(lens_phase, which=1)
        self.update_slm(which=1, wait=0.5)
        # Get the image from the camera
        image = self.get_bench_image()
        image = torch.abs(image - self.background_image)
        return image, lens_phase

    def classifier_forward(self, sample, target):
        # reshape the sample into [b,1,h,w]
        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0).unsqueeze(0)
        elif len(sample.shape) == 3:
            sample = sample.unsqueeze(1)
        batch = (sample, target)
        prediction, target = self.classifier.shared_step(batch)
        return prediction, target

    def forward(self, samples, slm_samples, classifier_targets):
        # Run the simulation forward
        simulation_outputs = self.dom_forward(samples)

        # Run the benchtop forward
        bench_image, phases = self.bench_forward(slm_samples)

        # Run the classifier forward
        if self.classifier_image == 'bench':
            classifier_image = bench_image
        elif self.classifier_image == 'sim':
            classifier_image = simulation_outputs['images']
        elif self.classifier_image == 'ideal':
            classifier_image = samples.abs()
        else:
            raise ValueError(f"Classifier image {self.classifier_image} not supported")

        classifier_output, classifier_target = self.classifier_forward(classifier_image, classifier_targets)

        return {'simulation_outputs': simulation_outputs,
                'bench_image': bench_image,
                'phases': phases,
                'classifier_output': classifier_output,
                'classifier_target': classifier_target}

    #-----------------------------------------
    # Initialize: Training utilities
    #-----------------------------------------

    def on_train_start(self):
        self.upload_benign_image(which=0)

    def on_train_epoch_end(self):
        # Get the phases from the simulation modulator
        phases = self.get_lens_from_dom()
        # Get the current epoch
        epoch = self.current_epoch
        # Save the lens phase
        torch.save(phases, os.path.join(self.paths['path_lens_phase'], f'lens_phase_{epoch:04d}.pt'))

    def shared_step(self, batch):
        samples, slm_samples, classifier_targets = batch

        # Run the forward pass
        outputs = self(samples, slm_samples, classifier_targets)

        # Parse the outputs
        simulation_outputs = outputs['simulation_outputs']
        bench_image = outputs['bench_image']
        phases = outputs['phases']
        classifier_output = outputs['classifier_output']
        classifier_target = outputs['classifier_target']

        return {'simulation_outputs': simulation_outputs,
                'bench_image': bench_image,
                'phases': phases,
                'classifier_output': classifier_output,
                'classifier_target': classifier_target,
                }

    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        # Get the loss
        loss, sim_to_ideal, sim_to_bench, bench_to_ideal, classifier_loss = self.objective(outputs, batch)
        # Log the loss
        self.log('loss_train', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('sim_to_ideal_train', sim_to_ideal, on_step=False, on_epoch=True, sync_dist=True)
        self.log('sim_to_bench_train', sim_to_bench, on_step=False, on_epoch=True, sync_dist=True)
        self.log('bench_to_ideal_train', bench_to_ideal, on_step=False, on_epoch=True, sync_dist=True)
        self.log('classifier_loss_train', classifier_loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        # Get the loss
        loss, sim_to_ideal, sim_to_bench, bench_to_ideal, classifier_loss = self.objective(outputs, batch)
        # Log the loss
        self.log('loss_val', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('sim_to_ideal_val', sim_to_ideal, on_step=False, on_epoch=True, sync_dist=True)
        self.log('sim_to_bench_val', sim_to_bench, on_step=False, on_epoch=True, sync_dist=True)
        self.log('bench_to_ideal_val', bench_to_ideal, on_step=False, on_epoch=True, sync_dist=True)
        self.log('classifier_loss_val', classifier_loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

#-----------------------------------------
# Initialize: Select model utility
#-----------------------------------------

def select_model(params):
    if params['model'] == 'classifier':
        model = Classifier(params)
    elif params['model'] == 'cooperative':
        model = CooperativeOpticalModel(params)
    else:
        raise ValueError("Model not supported")
    return model

if __name__ == "__main__":

    sys.path.append('../')
    params = yaml.load(open('../../config.yaml', 'r'), Loader=yaml.FullLoader)
    params['paths']['path_root'] = '../../'
    model = CooperativeOpticalModel(params)
    from datamodule.datamodule import select_data

    datamodule = select_data(params)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    batch = next(iter(train_dataloader))
    sample, slm_sample, target = batch

    outputs = model(sample, slm_sample, target)
