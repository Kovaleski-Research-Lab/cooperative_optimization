

import os
import sys
import torch
import argparse
import numpy as np
from torchvision.datasets import MNIST

sys.path.append('../')

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Subsample the MNIST dataset')
    parser.add_argument('--path_root', type=str, default='../', help='Path to the root directory of the project')
    parser.add_argument('--path_data', type=str, default='../data/', help='Path to the data directory')
    parser.add_argument('--path_output', type=str, default='../data/', help='Path to the output directory')
    parser.add_argument('--samples_per_class', type=int, default=100, help='Number of samples to keep per class')
    parser.add_argument('--classes', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='List of classes to keep')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--overwrite', type=bool, default=False, help='Overwrite the output file if it exists')
    parser.add_argument('--filename', type=str, default='mnist_subsampled.pt', help='Name of the output file')
    args = parser.parse_args()

    
    # Load the MNIST dataset
    trainset = MNIST(root=args.path_data, train=True, download=True)
    testset = MNIST(root=args.path_data, train=False, download=True)

    # Get the classes
    classes = args.classes

    # Get data
    train_data = trainset.data.numpy()
    train_labels = trainset.targets.numpy()

    test_data = testset.data.numpy()
    test_labels = testset.targets.numpy()

    # Get the indices of the classes specified
    train_indices = [np.where(train_labels == class_idx)[0] for class_idx in args.classes]
    test_indices = [np.where(test_labels == class_idx)[0] for class_idx in args.classes]

    # Get the number of samples per class
    samples_per_class = args.samples_per_class

    # Set the seed
    np.random.seed(args.seed)

    # Randomly select samples_per_class samples from each class
    train_indices = [np.random.choice(indices, samples_per_class, replace=False) for indices in train_indices]
    test_indices = [np.random.choice(indices, samples_per_class, replace=False) for indices in test_indices]

    # Concatenate the indices
    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    # Subsample the data
    train_data = train_data[train_indices]
    train_labels = train_labels[train_indices]

    test_data = test_data[test_indices]
    test_labels = test_labels[test_indices]

    # Convert to tensors
    train_data = torch.tensor(train_data)
    train_labels = torch.tensor(train_labels)

    test_data = torch.tensor(test_data)
    test_labels = torch.tensor(test_labels)

    # Save the subsampled data
    output_file = os.path.join(args.path_output, args.filename)

    if os.path.exists(output_file) and not args.overwrite:
        raise FileExistsError("Output file already exists. Use --overwrite to overwrite the file.")

    torch.save({'train_data': train_data, 'train_labels': train_labels, 'test_data': test_data, 'test_labels': test_labels}, output_file)
