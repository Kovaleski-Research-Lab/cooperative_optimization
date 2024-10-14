
# Create data split for cooperative optimization.

# The datasets will be a variable number of random samples taken from 
# each class of the MNIST data set, i.e., 10 random samples from each class.
# We will assume that the train/valid splits will be handled by the datamodule.
# While we will want to resize and binarize the images, we will not want to
# save these larger files to disk. Instead, we will save the originals and
# call transformations on the fly.

import os
import torch
import argparse
import numpy as np
from torchvision.datasets import MNIST

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Create data splits for cooperative optimization.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save the data.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to take from each class.')
    args = parser.parse_args()

    # Seed
    torch.manual_seed(123)

    # Load the MNIST data
    data_dir = args.data_dir
    mnist_data = MNIST(data_dir, train=True, download=False).data
    mnist_labels = MNIST(data_dir, train=True, download=False).targets

    # Create the data split
    data_split = []
    labels_split = []
    for i in range(10):
        idx = (mnist_labels == i).nonzero().squeeze()
        idx = idx[torch.randperm(idx.size(0))[:args.num_samples]]
        data_split.append(mnist_data[idx])
        labels_split.append(mnist_labels[idx])

    # Turn the lists into tensors
    data_split = torch.stack(data_split)
    labels_split = torch.stack(labels_split)

    # Combine the first two dimensions
    data_split = data_split.view(-1, 1, 28, 28)
    labels_split = labels_split.view(-1)

    # Create a dictionary to save the data split
    data_split = {'data': data_split, 'labels': labels_split}

    # Save the data split
    torch.save(data_split, os.path.join(data_dir, f'data_split_{args.num_samples:05d}.pt'))

