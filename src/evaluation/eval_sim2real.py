import os
import csv
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from diffractive_optical_model.diffractive_optical_model import DOM


sys.path.append('/develop/code/cooperative_optimization/src')
sys.path.append('../')
from datamodule.datamodule import select_data
from models.models import Sim2Real


def plot_images(model, data_files):

    for i,file in enumerate(data_files):
        # Load the data
        data = torch.load(file, weights_only=True)
        sample = data['resampled_sample']
        bench_image = data['bench_image']
        original_sim_output = data['sim_output']

        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0).unsqueeze(0)
        elif len(sample.shape) == 3:
            sample = sample.unsqueeze(0)

        # Forward pass
        sim_output = model.forward(sample.cpu())

        if i == 0:
            # Create the figure
            fig, ax = plt.subplots(1,4, figsize=(20,5))
            im0 = ax[0].imshow(sample.squeeze().cpu().detach().numpy())
            ax[0].set_title('Object')
            ax[0].axis('off')
            im1 = ax[1].imshow(bench_image.squeeze().cpu().detach().numpy())
            ax[1].set_title('Bench Image')
            ax[1].axis('off')
            im2 = ax[2].imshow(original_sim_output.squeeze().cpu().detach().abs().numpy()**2)
            ax[2].set_title('Simulated Image')
            ax[2].axis('off')
            im3 = ax[3].imshow(sim_output.squeeze().cpu().detach().abs().numpy()**2)
            ax[3].set_title('New Simulated Image')
            ax[3].axis('off')
        else:
            im0.set_data(sample.squeeze().cpu().detach().numpy())
            im1.set_data(bench_image.squeeze().cpu().detach().numpy())
            im2.set_data(original_sim_output.squeeze().cpu().detach().abs().numpy()**2)
            im3.set_data(sim_output.squeeze().cpu().detach().abs().numpy()**2)

        plt.pause(1)


def plot_differences(model, data_files):

    for i,file in enumerate(data_files):
        # Load the data
        data = torch.load(file, weights_only=True)
        sample = data['resampled_sample']
        bench_image = data['bench_image'].squeeze().cpu().detach()
        original_sim_output = data['sim_output'].abs().detach().cpu()**2

        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0).unsqueeze(0)
        elif len(sample.shape) == 3:
            sample = sample.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            sim_output = model.forward(sample.cpu()).detach().abs().cpu()**2

        bench_original_sim_diff = torch.abs(bench_image - original_sim_output).squeeze()
        bench_new_sim_diff = torch.abs(bench_image - sim_output).squeeze()
        bench_image = bench_image.squeeze()
        original_sim_output = original_sim_output.squeeze()
        sim_output = sim_output.squeeze()

        if i == 0:
            fig, ax = plt.subplots(2,3, figsize=(15,10))
            im0 = ax[0][0].imshow(bench_image)
            ax[0][0].set_title('Bench Image')
            ax[0][0].axis('off')

            im1 = ax[0][1].imshow(original_sim_output)
            ax[0][1].set_title('Original Simulated Image')
            ax[0][1].axis('off')

            im2 = ax[0][2].imshow(bench_original_sim_diff)
            ax[0][2].set_title('Diff Original')
            ax[0][2].axis('off')

            im3 = ax[1][0].imshow(bench_image)
            ax[1][0].set_title('Bench Image')
            ax[1][0].axis('off')
            
            im4 = ax[1][1].imshow(sim_output)
            ax[1][1].set_title('New Simulated Image')
            ax[1][1].axis('off')

            im5 = ax[1][2].imshow(bench_new_sim_diff)
            ax[1][2].set_title('Diff New')
            ax[1][2].axis('off')
        else:
            im0.set_data(bench_image)
            im1.set_data(original_sim_output)
            im2.set_data(bench_original_sim_diff)
            im3.set_data(bench_image)
            im4.set_data(sim_output)
            im5.set_data(bench_new_sim_diff)

        plt.pause(1)


def get_calibration_layer(model, save=False, path_save=None):

    # Get the calibration layer
    #amplitude = model.dom.layers[1].modulator.get_amplitude().squeeze().cpu().detach().numpy()
    #phase = model.dom.layers[1].modulator.get_phase().squeeze().cpu().detach().numpy()
    amplitude = model.dom.layers[1].modulator.optimizeable_amplitude.squeeze().cpu().detach().numpy()
    phase = model.dom.layers[1].modulator.optimizeable_phase.squeeze().cpu().detach().numpy()
    calibration_layer = amplitude * np.exp(1j*phase)

    if save:
        torch.save(amplitude, os.path.join(path_save, 'calibration_amplitude.pt'))
        torch.save(phase, os.path.join(path_save, 'calibration_phase.pt'))
        torch.save(calibration_layer, os.path.join(path_save, 'calibration_layer.pt'))

    return calibration_layer


def plot_calibration_layer(layer):

    amplitude = np.abs(layer)
    phase = np.angle(layer)

    fig,ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].imshow(amplitude)
    ax[0].set_title('Amplitude')
    ax[0].axis('off')

    ax[1].imshow(phase, cmap='hsv')
    ax[1].set_title('Phase')
    ax[1].axis('off')

    plt.show()


def run_model(model, data_files, save=False, path_save=None):

    for i,file in enumerate(tqdm(data_files)):
        # Load the data
        data = torch.load(file, weights_only=True)
        sample = data['resampled_sample'].cuda()
        bench_image = data['bench_image'].squeeze().cpu().detach()
        original_sim_output = data['sim_output'].detach().cpu()
        target = data['target']

        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0).unsqueeze(0)
        elif len(sample.shape) == 3:
            sample = sample.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            sim_output = model.forward(sample).detach().abs().cpu()**2

        if save:
            split = file.split('/')[-1]
            split = split.split('_')
            split[0] = 'sim2real'
            if 'train' in file:
                temp = split[-1]
                split[1] = 'train'
                split.append(temp)
            elif 'valid' in file:
                temp = split[-1]
                split[1] = 'valid'
                split.append(temp)
            split = '_'.join(split)
            data = {'resampled_sample': sample, 'bench_image': bench_image, 'sim2real_output': sim_output, 'sim_output': original_sim_output, 'target': target}
            torch.save(data, os.path.join(path_save, split))

if __name__ == "__main__":

    pl.seed_everything(123)
    # Experiment path
    versions = ['version_5']
    for version in versions:
        path_experiment = '/develop/results/sim2real/' + version
        path_checkpoint = os.path.join(path_experiment, 'checkpoints', 'last.ckpt')
        path_config = os.path.join(path_experiment, 'config.yaml')

        # Load the config file
        config = yaml.load(open(path_config, 'r'), Loader=yaml.FullLoader)

        # Load the model from the checkpoint
        model = Sim2Real.load_from_checkpoint(path_checkpoint, params=config, strict=True).cuda()

        # I don't have a good datamodule for this eval, so we will load the images
        # manually.
        #path_data = os.path.join(config['paths']['path_root'], config['paths']['path_data'])
        path_data = os.path.join('/develop/data/baseline')
        files = os.listdir(path_data)
        files.sort()
        train_files = [os.path.join(path_data, f) for f in files if 'train' in f]
        valid_files = [os.path.join(path_data, f) for f in files if 'valid' in f]

        # Plot the calibration layer
        calibration_layer = get_calibration_layer(model, save=True, path_save=path_experiment)
        #plot_calibration_layer(calibration_layer)

        # Plot the images
        #plot_images(model, train_files+valid_files)
        #plot_differences(model, train_files+valid_files)

        # Save the new simulated images
        os.makedirs(os.path.join('/develop/data/sim2real/' + version), exist_ok=True)
        run_model(model, train_files+valid_files, save=True, path_save = '/develop/data/sim2real/' + version)

