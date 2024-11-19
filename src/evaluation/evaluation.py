import os
import csv
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffractive_optical_model.diffractive_optical_model import DOM

from torchmetrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall

from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import mean_squared_error as MSE
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
from joblib import Parallel, delayed

sys.path.append('../')
from datamodule.datamodule import select_data
from models.models import Classifier

def load_images(path_images):
    train_images = []
    valid_images = []
    train_labels = []
    valid_labels = []
    filenames = os.listdir(path_images)
    for filename in tqdm(filenames, desc='Loading images'):
        data = torch.load(os.path.join(path_images, filename), weights_only=True)
        bench_image = data['bench_image'].squeeze().detach().cpu().numpy()
        sim_image = data['sim_output'].squeeze().detach().cpu().numpy()
        ideal_image = data['resampled_sample'].squeeze().detach().cpu().numpy()
        target = data['target'].squeeze().detach().cpu()
        target = torch.argmax(target, dim=-1).numpy()
        if 'train' in filename:
            train_images.append((bench_image, sim_image, ideal_image, target))
            train_labels.append(target)
        elif 'valid' in filename:
            valid_images.append((bench_image, sim_image, ideal_image, target))
            valid_labels.append(target)
        else:
            raise ValueError('Invalid filename')

    return train_images, valid_images, train_labels, valid_labels

def plot_label_histogram(train_labels, valid_labels, save=False, path_save=None):

    fig, ax = plt.subplots(2,1, figsize=(5,5))

    style = {'facecolor': '#bbccee', 'edgecolor': '#222255', 'linewidth': 1}

    n, bins_train, patches = ax[0].hist(train_labels, bins=10, label='Train', **style)
    ax[0].set_title('Train labels')
    ax[0].set_xlabel('Label')
    ax[0].set_ylabel('Count')

    ax[1].hist(valid_labels, bins=10, label='Valid', **style)
    ax[1].set_title('Valid labels')
    ax[1].set_xlabel('Label')
    ax[1].set_ylabel('Count')


    bin_centers = 0.5 * np.diff(bins_train) + bins_train[:-1]

    for a in ax.flatten():
        a.set_xticks(bin_centers, [i for i in range(0,10)])

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(os.path.join(path_save, 'label_histogram.pdf'))
        plt.close('all')

def plot_images(train_images, valid_images):
    images = train_images + valid_images
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    im0 = ax[0].imshow(images[0][2])
    ax[0].set_title('Ideal image')
    im1 = ax[1].imshow(images[0][1])
    ax[1].set_title('Simulation output')
    im2 = ax[2].imshow(images[0][0])
    ax[2].set_title('Resampled sample')

    for b, s, i, t in images:
        im0.set_data(i)
        im1.set_data(s)
        im2.set_data(b)
        fig.suptitle(f'Label: {t}')
        plt.pause(1.0)

def plot_image_differences(train_images, valid_images):
    images = train_images + valid_images

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ideal_image = images[0][2]
    bench_image = images[0][0]
    sim_image = images[0][1]

    diff_bench = np.abs(ideal_image - bench_image)
    diff_sim = np.abs(ideal_image - sim_image)
    diff_bench_sim = np.abs(bench_image - sim_image)

    im0 = ax[0].imshow(diff_bench)
    ax[0].set_title('Ideal - Bench')
    im1 = ax[1].imshow(diff_sim)
    ax[1].set_title('Ideal - Sim')
    im2 = ax[2].imshow(diff_bench_sim)
    ax[2].set_title('Bench - Sim')

    for b, s, i, t in images:

        i = normalize_image_mean_std(i)
        s = normalize_image_mean_std(s)
        b = normalize_image_mean_std(b)

        diff_bench = i - b
        diff_sim = i - s
        diff_bench_sim = b - s

        min_diff_bench = np.min(diff_bench)
        max_diff_bench = np.max(diff_bench)

        min_diff_sim = np.min(diff_sim)
        max_diff_sim = np.max(diff_sim)

        min_diff_bench_sim = np.min(diff_bench_sim)
        max_diff_bench_sim = np.max(diff_bench_sim)

        im0.set_data(diff_bench)
        im0.set(clim=(min_diff_bench, max_diff_bench))
        im1.set_data(diff_sim)
        im1.set(clim=(min_diff_sim, max_diff_sim))
        im2.set_data(diff_bench_sim)
        im2.set(clim=(min_diff_bench_sim, max_diff_bench_sim))

        #ideal_bench = np.zeros((i.shape[0], i.shape[1], 3))
        #ideal_bench[:,:,0] = i
        #ideal_bench[:,:,2] = b

        #ideal_sim = np.zeros((i.shape[0], i.shape[1], 3))
        #ideal_sim[:,:,0] = i
        #ideal_sim[:,:,2] = s

        #bench_sim = np.zeros((i.shape[0], i.shape[1], 3))
        #bench_sim[:,:,0] = b
        #bench_sim[:,:,2] = s

        #im0.set_data(ideal_bench)
        #im1.set_data(ideal_sim)
        #im2.set_data(bench_sim)
        plt.pause(0.2)


def plot_normalized_images(train_images, valid_images):
    images = train_images + valid_images
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    im0 = ax[0].imshow(images[0][2])
    ax[0].set_title('Ideal image')
    im1 = ax[1].imshow(images[0][1])
    ax[1].set_title('Simulation output')
    im2 = ax[2].imshow(images[0][0])
    ax[2].set_title('Resampled sample')

    for b, s, i, t in images:
        i = normalize_image_mean_std(i)
        s = normalize_image_mean_std(s)
        b = normalize_image_mean_std(b)
        im0.set_data(i)
        im1.set_data(s)
        im2.set_data(b)
        # Auto scale the images
        im0.set_clim(np.min(i), np.max(i))
        im1.set_clim(np.min(s), np.max(s))
        im2.set_clim(np.min(b), np.max(b))
        fig.suptitle(f'Label: {t}')
        plt.pause(1.0)

def normalize_image_linear(image):
    min_value = np.min(image)
    max_value = np.max(image)

    new_min = 0
    new_max = 1

    normalized_image = ((image - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
    return normalized_image


def normalize_image_mean_std(image):
    mean = np.mean(image)
    std = np.std(image)

    normalized_image = (image - mean) / std
    return normalized_image


def calculate_image_metrics(bench_image, sim_image, ideal_image):
    psnr = []
    mse = []
    ssim = []
    max = []
    min = []
    mean = []

    ideal_image = torch.from_numpy(np.copy(ideal_image)).squeeze().cuda()
    bench_image = torch.from_numpy(np.copy(bench_image)).squeeze().cuda()
    sim_image = torch.from_numpy(np.copy(sim_image)).squeeze().cuda()

    psnr.append(PSNR(ideal_image, bench_image))
    psnr.append(PSNR(ideal_image, sim_image))
    psnr.append(PSNR(bench_image, sim_image))

    mse.append(MSE(ideal_image, bench_image))
    mse.append(MSE(ideal_image, sim_image))
    mse.append(MSE(bench_image, sim_image))

    # For SSIM, images need to be in (B, C, H, W) format
    bench_image = bench_image.unsqueeze(0).unsqueeze(0)
    sim_image = sim_image.unsqueeze(0).unsqueeze(0)
    ideal_image = ideal_image.unsqueeze(0).unsqueeze(0)

    ssim.append(SSIM(ideal_image, bench_image))
    ssim.append(SSIM(ideal_image, sim_image))
    ssim.append(SSIM(bench_image, sim_image))

    max.append(torch.max(ideal_image).detach().cpu().numpy())
    max.append(torch.max(bench_image).detach().cpu().numpy())
    max.append(torch.max(sim_image).detach().cpu().numpy())

    min.append(torch.min(ideal_image).detach().cpu().numpy())
    min.append(torch.min(bench_image).detach().cpu().numpy())
    min.append(torch.min(sim_image).detach().cpu().numpy())

    mean.append(torch.mean(ideal_image).detach().cpu().numpy())
    mean.append(torch.mean(bench_image).detach().cpu().numpy())
    mean.append(torch.mean(sim_image).detach().cpu().numpy())

    return psnr, ssim, mse, max, min, mean

def compare_images(train_images, valid_images):
    psnr = {'ideal_to_bench': [], 'ideal_to_sim': [], 'bench_to_sim': []}
    ssim = {'ideal_to_bench': [], 'ideal_to_sim': [], 'bench_to_sim': []}
    mse = {'ideal_to_bench': [], 'ideal_to_sim': [], 'bench_to_sim': []}
    max = {'bench': [], 'sim': []}
    min = {'bench': [], 'sim': []}
    mean = {'bench': [], 'sim': []}

    results = Parallel(n_jobs=2)(delayed(calculate_image_metrics)(bench_image, sim_image, ideal_image) for bench_image, sim_image, ideal_image, target in tqdm(train_images))
    for p, s, m, ma, mi, me in results:
        psnr['ideal_to_bench'].append(p[0].detach().cpu().numpy())
        psnr['ideal_to_sim'].append(p[1].detach().cpu().numpy())
        psnr['bench_to_sim'].append(p[2].detach().cpu().numpy())

        mse['ideal_to_bench'].append(m[0].detach().cpu().numpy())
        mse['ideal_to_sim'].append(m[1].detach().cpu().numpy())
        mse['bench_to_sim'].append(m[2].detach().cpu().numpy())

        ssim['ideal_to_bench'].append(s[0].detach().cpu().numpy())
        ssim['ideal_to_sim'].append(s[1].detach().cpu().numpy())
        ssim['bench_to_sim'].append(s[2].detach().cpu().numpy())

        max['bench'].append(ma[1])
        max['sim'].append(ma[2])

        min['bench'].append(mi[1])
        min['sim'].append(mi[2])

        mean['bench'].append(me[1])
        mean['sim'].append(me[2])

    results = Parallel(n_jobs=4)(delayed(calculate_image_metrics)(bench_image, sim_image, ideal_image) for bench_image, sim_image, ideal_image, target in tqdm(valid_images))
    for p, s, m, ma, mi, me in results:
        psnr['ideal_to_bench'].append(p[0].detach().cpu().numpy())
        psnr['ideal_to_sim'].append(p[1].detach().cpu().numpy())
        psnr['bench_to_sim'].append(p[2].detach().cpu().numpy())

        mse['ideal_to_bench'].append(m[0].detach().cpu().numpy())
        mse['ideal_to_sim'].append(m[1].detach().cpu().numpy())
        mse['bench_to_sim'].append(m[2].detach().cpu().numpy())

        ssim['ideal_to_bench'].append(s[0].detach().cpu().numpy())
        ssim['ideal_to_sim'].append(s[1].detach().cpu().numpy())
        ssim['bench_to_sim'].append(s[2].detach().cpu().numpy())

        max['bench'].append(ma[1])
        max['sim'].append(ma[2])
        min['bench'].append(mi[1])
        min['sim'].append(mi[2])

        mean['bench'].append(me[1])
        mean['sim'].append(me[2])

    return psnr, ssim, mse, max, min, mean

def plot_image_comparisons(psnr, ssim, mse, max, min, mean, save=False, path_save=None):
    plot_comparisons(psnr, ssim, mse, save=save, path_save=path_save)
    plot_count_statistics(max, min, mean, save=save, path_save=path_save)
    return

def plot_comparisons(psnr, ssim, mse, save=False, path_save=None):
    labels = ['Simulation to ideal', 'Bench to ideal', 'Simulation to bench']
    fig, ax = plt.subplots(3, 1, figsize=(5,8))

    vp0 = ax[0].violinplot([mse['ideal_to_sim'], mse['ideal_to_bench'], mse['bench_to_sim']], points = 1000, showmeans=True)
    ax[0].set_xticks([i+1 for i in range(len(labels))], labels)
    ax[0].set_ylabel(r'MSE $\downarrow$')

    vp1 = ax[1].violinplot([psnr['ideal_to_sim'], psnr['ideal_to_bench'], psnr['bench_to_sim']], points = 1000, showmeans=True)
    ax[1].set_xticks([i+1 for i in range(len(labels))], labels)
    ax[1].set_ylabel(r'PSNR $\uparrow$')

    vp2 = ax[2].violinplot([ssim['ideal_to_sim'], ssim['ideal_to_bench'], ssim['bench_to_sim']], points = 1000, showmeans=True)
    ax[2].set_xticks([i+1 for i in range(len(labels))], labels)
    ax[2].set_ylabel(r'SSIM $\uparrow$')
    
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(path_save, 'image_comparisons.pdf'))
        plt.close('all')
    else:
        plt.show()
    
def plot_count_statistics(max, min, mean, save=False, path_save=None):
    labels = ['Bench', 'Simulation']

    fig, ax = plt.subplots(3,1, figsize=(5,4))

    vp0 = ax[0].violinplot([max['bench'], max['sim']], points = 1000, showmeans=True)
    ax[0].set_xticks([i+1 for i in range(len(labels))], labels)
    ax[0].set_ylabel('Max value')

    vp1 = ax[1].violinplot([min['bench'], min['sim']], points = 1000, showmeans=True)
    ax[1].set_xticks([i+1 for i in range(len(labels))], labels)
    ax[1].set_ylabel('Min value')

    vp2 = ax[2].violinplot([mean['bench'], mean['sim']], points = 1000, showmeans=True)
    ax[2].set_xticks([i+1 for i in range(len(labels))], labels)
    ax[2].set_ylabel('Mean value')

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(path_save, 'image_count_statistics.pdf'))
        plt.close('all')
    else:
        plt.show()

def load_classifier_checkpoint(checkpoint_path):
    classifier = Classifier.load_from_checkpoint(os.path.join(checkpoint_path, 'checkpoints', 'last.ckpt')).double().cpu()
    return classifier

def validate_classifier_weights(classifier, checkpoint_path):
    checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoints', 'last.ckpt'), weights_only=True)
    state_dict = checkpoint['state_dict']
    pass

def load_loss_metrics(checkpoint_path):
    filename = os.path.join(checkpoint_path, 'logs', 'metrics.csv')
    metrics = {}
    key_list = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            if i == 0:
                for header in row:
                    metrics[header] = []
                key_list = list(metrics.keys())
            else:
                for j, value in enumerate(row):
                    metrics[key_list[j]].append(value)


    metrics['epoch'] = np.unique(np.asarray(metrics['epoch'], dtype=int))
    for key in key_list:
        if key != 'epoch':
            metrics[key] = np.asarray([float(value) for value in metrics[key] if value != ''], dtype=float)

    return metrics

def plot_loss_metrics(metrics, save=False, path_save=None):
    # Get the metrics
    epochs = metrics['epoch']
    loss_train = metrics['loss_train']
    loss_valid = metrics['loss_val']

    # Create some ticks for the x axis
    ticks = np.arange(0, len(epochs), 2)
    tick_labels = ticks

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.plot(epochs, loss_train, label='Train loss')
    ax.plot(epochs, loss_valid, label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy loss')

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.legend()
    if save:
        fig.savefig(os.path.join(path_save, 'loss_metrics.pdf'))
    else:
        plt.show()

def eval_classifier(classifier, images):
    feature_vectors = {'bench': [], 'sim': [], 'ideal': []}
    predictions = {'bench': [], 'sim': [], 'ideal': []}

    with torch.no_grad():
        for bench_image, sim_image, ideal_image, target in tqdm(images, desc='Evaluating classifier'):
            target = torch.from_numpy(target).unsqueeze(0).double().cuda()
            bench_image = torch.from_numpy(bench_image).unsqueeze(0).unsqueeze(0)
            bench_image = torch.cat([bench_image, bench_image, bench_image], dim=1).double().cuda()
            sim_image = torch.from_numpy(sim_image).unsqueeze(0).unsqueeze(0)
            sim_image = torch.cat([sim_image, sim_image, sim_image], dim=1).double().cuda()
            ideal_image = torch.from_numpy(ideal_image).unsqueeze(0).unsqueeze(0)
            ideal_image = torch.cat([ideal_image, ideal_image, ideal_image], dim=1).double().cuda()

            feature_vector_bench = classifier.feature_extractor(bench_image)
            feature_vector_sim = classifier.feature_extractor(sim_image)
            feature_vector_ideal = classifier.feature_extractor(ideal_image)

            prediction_bench = torch.argmax(classifier(bench_image), dim=-1)
            prediction_sim = torch.argmax(classifier(sim_image), dim=-1)
            prediction_ideal = torch.argmax(classifier(ideal_image), dim=-1)

            feature_vectors['bench'].append(feature_vector_bench.detach().cpu().numpy())
            feature_vectors['sim'].append(feature_vector_sim.detach().cpu().numpy())
            feature_vectors['ideal'].append(feature_vector_ideal.detach().cpu().numpy())

            predictions['bench'].append([prediction_bench.detach().cpu(), torch.argmax(target, dim=-1).cpu()])
            predictions['sim'].append([prediction_sim.detach().cpu(), torch.argmax(target, dim=-1).cpu()])
            predictions['ideal'].append([prediction_ideal.detach().cpu(), torch.argmax(target, dim=-1).cpu()])

    return feature_vectors, predictions


def save_classifier_results(path_classifier_eval, train_feature_vectors, train_predictions, valid_feature_vectors, valid_predictions):
    torch.save(train_feature_vectors, os.path.join(path_classifier_eval, 'train_feature_vectors.pt'))
    torch.save(train_predictions, os.path.join(path_classifier_eval, 'train_predictions.pt'))
    torch.save(valid_feature_vectors, os.path.join(path_classifier_eval, 'valid_feature_vectors.pt'))
    torch.save(valid_predictions, os.path.join(path_classifier_eval, 'valid_predictions.pt'))

def load_classifier_results(path_classifier_eval):
    train_feature_vectors = torch.load(os.path.join(path_classifier_eval, 'train_feature_vectors.pt'))
    train_predictions = torch.load(os.path.join(path_classifier_eval, 'train_predictions.pt'))
    valid_feature_vectors = torch.load(os.path.join(path_classifier_eval, 'valid_feature_vectors.pt'))
    valid_predictions = torch.load(os.path.join(path_classifier_eval, 'valid_predictions.pt'))

    return train_feature_vectors, train_predictions, valid_feature_vectors, valid_predictions

def calculate_confusion_matrices(train_predictions, valid_predictions):
    confmat = ConfusionMatrix(task = 'multiclass', num_classes=10)

    cfms = {}
    ideal_train = []
    bench_train = []
    sim_train = []

    ideal_valid = []
    bench_valid = []
    sim_valid = []

    for pred, target in train_predictions['ideal']:
        ideal_train.append([pred, target.cpu()])
    ideal_train = torch.tensor(ideal_train)
    for pred, target in train_predictions['bench']:
        bench_train.append([pred, target.cpu()])
    bench_train = torch.tensor(bench_train)
    for pred, target in train_predictions['sim']:
        sim_train.append([pred, target.cpu()])
    sim_train = torch.tensor(sim_train)

    for pred, target in valid_predictions['ideal']:
        ideal_valid.append([pred, target.cpu()])
    ideal_valid = torch.tensor(ideal_valid)
    for pred, target in valid_predictions['bench']:
        bench_valid.append([pred, target.cpu()])
    bench_valid = torch.tensor(bench_valid)
    for pred, target in valid_predictions['sim']:
        sim_valid.append([pred, target.cpu()])
    sim_valid = torch.tensor(sim_valid)

    cfms['cfm_ideal_train'] = confmat(ideal_train[:,0], ideal_train[:,1])
    cfms['cfm_bench_train'] = confmat(bench_train[:,0], bench_train[:,1])
    cfms['cfm_sim_train'] = confmat(sim_train[:,0], sim_train[:,1])

    cfms['cfm_ideal_valid'] = confmat(ideal_valid[:,0], ideal_valid[:,1])
    cfms['cfm_bench_valid'] = confmat(bench_valid[:,0], bench_valid[:,1])
    cfms['cfm_sim_valid'] = confmat(sim_valid[:,0], sim_valid[:,1])
    return cfms

def plot_confusion_matrics(confusion_matrices, save=False, path_save=None):
    for k,value in confusion_matrices.items():
        df = pd.DataFrame(value, index = [i for i in range(10)], columns = [i for i in range(10)])
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        sns.heatmap(df, annot=True, ax=ax, square=True, cbar=False, cmap='Blues')
        ax.set_title(k)
        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(path_save, k + '.pdf'))
        else:
            plt.show()

def calculate_f1_scores(train_predictions, valid_predictions):
    f1_scores = {}
    f1 = F1Score(task='multiclass', num_classes=10)

    ideal_train = []
    bench_train = []
    sim_train = []

    ideal_valid = []
    bench_valid = []
    sim_valid = []

    for pred, target in train_predictions['ideal']:
        ideal_train.append([pred, target.cpu()])
    ideal_train = torch.tensor(ideal_train)
    for pred, target in train_predictions['bench']:
        bench_train.append([pred, target.cpu()])
    bench_train = torch.tensor(bench_train)
    for pred, target in train_predictions['sim']:
        sim_train.append([pred, target.cpu()])
    sim_train = torch.tensor(sim_train)

    for pred, target in valid_predictions['ideal']:
        ideal_valid.append([pred, target.cpu()])
    ideal_valid = torch.tensor(ideal_valid)
    for pred, target in valid_predictions['bench']:
        bench_valid.append([pred, target.cpu()])
    bench_valid = torch.tensor(bench_valid)
    for pred, target in valid_predictions['sim']:
        sim_valid.append([pred, target.cpu()])
    sim_valid = torch.tensor(sim_valid)

    f1_scores['f1_ideal_train'] = f1(ideal_train[:,0], ideal_train[:,1])
    f1_scores['f1_bench_train'] = f1(bench_train[:,0], bench_train[:,1])
    f1_scores['f1_sim_train'] = f1(sim_train[:,0], sim_train[:,1])

    f1_scores['f1_ideal_valid'] = f1(ideal_valid[:,0], ideal_valid[:,1])
    f1_scores['f1_bench_valid'] = f1(bench_valid[:,0], bench_valid[:,1])
    f1_scores['f1_sim_valid'] = f1(sim_valid[:,0], sim_valid[:,1])

    return f1_scores

def plot_feature_space(train_feature_vectors, valid_feature_vectors, train_predictions, valid_predictions, save=False, path_save=None):

    colors= [   '#a6cee3',
                '#1f78b4',
                '#b2df8a',
                '#33a02c',
                '#fb9a99',
                '#e31a1c',
                '#fdbf6f',
                '#ff7f00',
                '#cab2d6',
                '#6a3d9a']
    ideal_train = np.asarray([np.asarray(f[0]).squeeze() for f in train_feature_vectors['ideal']])
    bench_train = np.asarray([np.asarray(f[0]).squeeze() for f in train_feature_vectors['bench']])
    sim_train = np.asarray([np.asarray(f[0]).squeeze() for f in train_feature_vectors['sim']])

    ideal_valid = np.asarray([np.asarray(f[0]).squeeze() for f in valid_feature_vectors['ideal']])
    bench_valid = np.asarray([np.asarray(f[0]).squeeze() for f in valid_feature_vectors['bench']])
    sim_valid = np.asarray([np.asarray(f[0]).squeeze() for f in valid_feature_vectors['sim']])

    train_targets = torch.tensor(train_predictions['ideal'])[:,1].numpy()
    valid_targets = torch.tensor(valid_predictions['ideal'])[:,1].numpy()

    unique_targets = np.unique(train_targets)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(ideal_train)

    ideal_train_pca = pca.transform(ideal_train)
    bench_train_pca = pca.transform(bench_train)
    sim_train_pca = pca.transform(sim_train)

    fig,ax = plt.subplots(1,3, figsize=(15,5))
    for target in unique_targets:
        indices = np.where(train_targets == target)[0]
        bench_values = bench_train_pca[indices]
        sim_values = sim_train_pca[indices]
        ideal_values = ideal_train_pca[indices]
        color = colors[target]
        ax[0].scatter(ideal_values[:,0], ideal_values[:,1], color=color, label=f'{target}')
        ax[0].set_title('Ideal embeddings')
        ax[1].scatter(bench_values[:,0], bench_values[:,1], color=color, label=f'{target}')
        ax[1].set_title('Bench embeddings')
        ax[2].scatter(sim_values[:,0], sim_values[:,1], color=color, label=f'{target}')
        ax[2].set_title('Simulation embeddings')

    for a in ax.flatten():
        a.set_aspect('equal')
        a.legend(frameon=True, framealpha=1)
        a.set_xlim(-5.5, 5.5)
        a.set_ylim(-5.5, 5.5)

    plt.show()
    if save:
        fig.savefig(os.path.join(path_save, 'pca_embeddings.pdf'))
        plt.close('all')

    # UMAP
    umap_transform = umap.UMAP(n_neighbors=5, random_state=42).fit(ideal_train)

    ideal_train_umap = umap_transform.transform(ideal_train)
    bench_train_umap = umap_transform.transform(bench_train)
    sim_train_umap = umap_transform.transform(sim_train)

    fig,ax = plt.subplots(1,3, figsize=(15,5))
    for target in unique_targets:
        indices = np.where(train_targets == target)[0]
        bench_values = bench_train_umap[indices]
        sim_values = sim_train_umap[indices]
        ideal_values = ideal_train_umap[indices]
        color = colors[target]
        ax[0].scatter(ideal_values[:,0], ideal_values[:,1], color=color, label=f'{target}')
        ax[0].set_title('Ideal embeddings')
        ax[1].scatter(bench_values[:,0], bench_values[:,1], color=color, label=f'{target}')
        ax[1].set_title('Bench embeddings')
        ax[2].scatter(sim_values[:,0], sim_values[:,1], color=color, label=f'{target}')
        ax[2].set_title('Simulation embeddings')

    for a in ax.flatten():
        a.set_aspect('equal')
        a.legend(frameon=True, framealpha=1)
        a.set_xlim(-10, 15)
        a.set_ylim(-10, 15)

    plt.show()

    if save:
        fig.savefig(os.path.join(path_save, 'umap_embeddings.pdf'))
        plt.close('all')

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-dark-palette')
    checkpoint_path = '../../results/classifier_baseline_bench_resampled_sample/version_1/'
    path_classifier_eval = os.path.join(checkpoint_path, 'classifier_eval')
    os.makedirs(path_classifier_eval, exist_ok=True)
    baseline_classifier = load_classifier_checkpoint(checkpoint_path).cuda()
    #validate_classifier_weights(baseline_classifier, checkpoint_path)

    eval_images = False

    ## Load the images
    train_images, valid_images, train_labels, valid_labels = load_images('../../data/baseline/')

    # Plot label histograms
    plot_label_histogram(train_labels, valid_labels, save=True, path_save=path_classifier_eval)

    # Plot the images
    plot_images(train_images, valid_images)
    #plot_image_differences(train_images, valid_images)
    #plot_normalized_images(train_images, valid_images)
    exit()

    if eval_images:
        # Image comparisons
        psnr, ssim, mse, max, min, mean = compare_images(train_images, valid_images)
        plot_image_comparisons(psnr, ssim, mse, max, min, mean, save=True, path_save=path_classifier_eval)

    # Load the loss metrics
    metrics = load_loss_metrics(checkpoint_path)

    ## Plot the loss metrics
    plot_loss_metrics(metrics, save=True, path_save=path_classifier_eval)

    ## Evaluate the classifier
    #train_feature_vectors, train_predictions = eval_classifier(baseline_classifier, train_images)
    #valid_feature_vectors, valid_predictions = eval_classifier(baseline_classifier, valid_images)

    #save_classifier_results(path_classifier_eval, train_feature_vectors, train_predictions, valid_feature_vectors, valid_predictions)
    train_feature_vectors, train_predictions, valid_feature_vectors, valid_predictions = load_classifier_results(path_classifier_eval)

    # Classifier confusion matrices
    confusion_matrices = calculate_confusion_matrices(train_predictions, valid_predictions)

    plot_confusion_matrics(confusion_matrices, save=True, path_save=path_classifier_eval)

    # Calculate F1 scores
    f1_scores = calculate_f1_scores(train_predictions, valid_predictions)
    torch.save(f1_scores, os.path.join(path_classifier_eval, 'f1_scores.pt'))

    # Classifier feature spaces
    plot_feature_space(train_feature_vectors, valid_feature_vectors, train_predictions, valid_predictions, save=True, path_save=path_classifier_eval)
