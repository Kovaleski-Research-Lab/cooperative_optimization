#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
from loguru import logger
from typing import Optional
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

import sys
sys.path.append('../')
sys.path.append(os.path.join(os.path.dirname(__file__)))
import custom_transforms as ct

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



#--------------------------------
# Initialize: Baseline Bench
#--------------------------------

class Baseline_Bench_DataModule(LightningDataModule):
    def __init__(self, params:dict):
        super().__init__()
        logger.debug("Initializing Baseline_Bench_DataModule")
        self.params = params.copy()
        self.n_cpus = self.params['n_cpus']
        self.path_data = self.params['paths']['path_data']
        self.path_root = self.params['paths']['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        self.batch_size = self.params['batch_size']
        self.which_data = self.params['which_data']

    def initialize_cpus(self, n_cpus:int) -> None:
        # Make sure default number of cpus is not more than the system has
        if n_cpus > os.cpu_count(): # type: ignore
            n_cpus = 1
        self.n_cpus = n_cpus 
        logger.debug("Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        data_files = os.listdir(self.path_data)
        self.train_data = [os.path.join(self.path_data, f) for f in data_files if 'train' in f]
        self.valid_data = [os.path.join(self.path_data, f) for f in data_files if 'valid' in f]

        if stage == "fit" or stage is None:
            self.train_dataset = customDatasetBaseline(self.train_data, self.which_data)
            self.valid_dataset = customDatasetBaseline(self.valid_data, self.which_data)
        else:
            raise NotImplementedError("Stage {} not implemented".format(stage))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=False)

#--------------------------------
# Initialize: Custom dataset Baseline
#--------------------------------

class customDatasetBaseline(Dataset):
    def __init__(self, data, which_data):
        logger.debug("Initializing customDatasetBaseline")
        self.data_filenames = data
        self.which_data = which_data

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):

        data = torch.load(self.data_filenames[idx], weights_only=True)
        if self.which_data == 'resampled_sample':
            sample = data['resampled_sample']
        elif self.which_data == 'bench_image':
            sample = data['bench_image'].float()
        elif self.which_data == 'sim_output':
            sample = data['sim_output']
        else:
            raise NotImplementedError("Data {} not implemented".format(self.which_data))
        target = data['target']
        return sample.unsqueeze(0), target.squeeze()

#--------------------------------
# Initialize: MNIST Wavefront
#--------------------------------

class Wavefront_MNIST_DataModule(LightningDataModule):
    def __init__(self, params: dict, transform:str = "") -> None:
        super().__init__() 
        logger.debug("Initializing Wavefront_MNIST_DataModule")
        self.params = params.copy()
        self.Nx = self.params['Nxp']
        self.Ny = self.params['Nyp']
        self.n_cpus = self.params['n_cpus']
        self.path_data = self.params['paths']['path_data']
        self.path_root = self.params['paths']['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        logger.debug("Setting path_data to {}".format(self.path_data))
        self.batch_size = self.params['batch_size']
        self.data_split = self.params['data_split']
        self.train_percent = self.params['train_percent']
        self.initialize_transform()
        self.initialize_cpus(self.n_cpus)

    def initialize_transform(self) -> None:
        resize_row = self.params['resize_row']
        resize_col = self.params['resize_col']

        pad_x = int(torch.div((self.Nx - resize_row), 2, rounding_mode='floor'))
        pad_y = int(torch.div((self.Ny - resize_col), 2, rounding_mode='floor'))

        padding = (pad_y, pad_x, pad_y, pad_x)

        self.transform = transforms.Compose([
                transforms.Resize((resize_row, resize_col), antialias=True), # type: ignore
                transforms.Pad(padding),
                ct.Threshold(0.2),
                ct.WavefrontTransform(self.params['wavefront_transform'])])

    def initialize_cpus(self, n_cpus:int) -> None:
        # Make sure default number of cpus is not more than the system has
        if n_cpus > os.cpu_count(): # type: ignore
            n_cpus = 1
        self.n_cpus = n_cpus 
        logger.debug("Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self) -> None:
        MNIST(self.path_data, train=True, download=True)
        MNIST(self.path_data, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        logger.debug("Setup() with datasplit = {}".format(self.data_split))

        # Load the MNIST subset generated from /utils/create_datasplit.py
        # A dictionary with keys 'data' and 'labels'
        self.data_dict = torch.load(os.path.join(self.path_data, self.data_split), weights_only=True)
        self.total_data = self.data_dict['data']
        self.total_labels = self.data_dict['labels']

        num_samples = self.total_data.shape[0]
        num_train = int(num_samples * self.train_percent)

        # Split the data into train, valid
        # Generate random indices, and then select the first num_train for training
        idx = torch.randperm(num_samples)
        self.train_data = (self.total_data[idx[:num_train]], self.total_labels[idx[:num_train]])
        self.valid_data = (self.total_data[idx[num_train:]], self.total_labels[idx[num_train:]])

        if stage == "fit" or stage is None:
            self.train_dataset = customDatasetMNIST(self.train_data, self.transform)
            self.valid_dataset = customDatasetMNIST(self.valid_data, self.transform)
        else:
            raise NotImplementedError("Stage {} not implemented".format(stage))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=False)

#--------------------------------
# Initialize: Custom dataset MNIST
#--------------------------------

class customDatasetMNIST(Dataset):
    def __init__(self, data, transform_sample):
        logger.debug("Initializing customDatasetMNIST")
        self.samples, self.labels = data[0], data[1]
        self.transform_sample = transform_sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, target = self.samples[idx], self.labels[idx]
        if self.transform_sample is not None:
            sample = self.transform_sample(sample)

        sample = sample.abs().to(torch.complex128)
        slm_sample = (sample.abs() * 255).to(torch.uint8)
        target = torch.nn.functional.one_hot(target, num_classes=10)

        return sample, slm_sample, target

#--------------------------------
# Initialize: Select dataset
#--------------------------------

def select_data(params):
    if params['which'] == 'MNIST' :
        return Wavefront_MNIST_DataModule(params) 
    elif params['which'] == 'baseline_bench':
        return Baseline_Bench_DataModule(params)
    else:
        raise NotImplementedError("Dataset {} not implemented".format(params['which']))

#--------------------------------
# Initialize: Testing
#--------------------------------

if __name__=="__main__":
    import yaml
    import torch
    import matplotlib.pyplot as plt
    from pytorch_lightning import seed_everything
    seed_everything(1337)

    #Load config file   
    params = yaml.load(open('../../config.yaml'), Loader = yaml.FullLoader).copy()
    params['batch_size'] = 3
    params['model_id'] = "test_0"
    params['paths']['path_root'] = '../../'

    dm = select_data(params)
    dm.prepare_data()
    dm.setup(stage="fit")

    from IPython import embed; embed()
    #View some of the data
    images, target = next(iter(dm.train_dataloader()))


