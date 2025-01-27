#
import os
import numpy as np
import torch
import yaml
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.append('../')
from models.models import CooperativeOpticalModelRemoteSim2Real
from datamodule import datamodule
from pytorch_lightning import seed_everything
from loguru import logger

from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional import mean_squared_error as mse
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

def calculate_feature_metrics(features):

    mn = torch.min(features)
    mx = torch.max(features)
    me = torch.mean(features)
    return {
            'features': features,
            'mn': mn, 
            'mx': mx, 
            'me' :me
            }


def calculate_classifier_metrics(predictions, targets):
    targets = torch.argmax(targets, dim=-1)
    ce = torch.nn.functional.cross_entropy(predictions, targets)

    return {
            'predictions': predictions,
            'targets': targets,
            'ce': ce, 
            }

def calculate_image_metrics(images):

    ideal, bench, simulation = images

    # Make sure the images are in the correct format [1,1,w,h]
    while len(ideal.shape) < 4:
        ideal = ideal.unsqueeze(0)
    while len(bench.shape) < 4:
        bench = bench.unsqueeze(0)
    while len(simulation.shape) < 4:
        simulation = simulation.unsqueeze(0)


    ideal_bench_mse = mse(ideal, bench)
    ideal_simulation_mse = mse(ideal, simulation)
    bench_simulation_mse = mse(bench, simulation)

    ideal_bench_psnr = psnr(ideal, bench)
    ideal_simulation_psnr = psnr(ideal, simulation)
    bench_simulation_psnr = psnr(bench, simulation)

    ideal_bench_ssim = ssim(ideal, bench)
    ideal_simulation_ssim = ssim(ideal, simulation)
    bench_simulation_ssim = ssim(bench, simulation)

    
    mse_values = {'ideal_bench': ideal_bench_mse, 
           'ideal_simulation': ideal_simulation_mse, 
           'bench_simulation': bench_simulation_mse}
    
    psnr_values = {'ideal_bench': ideal_bench_psnr, 
            'ideal_simulation': ideal_simulation_psnr, 
            'bench_simulation': bench_simulation_psnr}
    
    ssim_values = {'ideal_bench': ideal_bench_ssim,
            'ideal_simulation': ideal_simulation_ssim, 
            'bench_simulation': bench_simulation_ssim}
    
    mn = {'ideal': torch.min(ideal), 'bench': torch.min(bench), 'simulation': torch.min(simulation)}
    mx = {'ideal': torch.max(ideal), 'bench': torch.max(bench), 'simulation': torch.max(simulation)}
    me = {'ideal': torch.mean(ideal), 'bench': torch.mean(bench), 'simulation': torch.mean(simulation)}

    return {
            'mse': mse_values, 
            'psnr': psnr_values, 
            'ssim': ssim_values, 
            'mn' : mn, 
            'mx' : mx, 
            'me' : me
            }

if __name__ == "__main__":

    config = yaml.load(open('../../config_coop_sim2real.yaml', 'r'), Loader=yaml.FullLoader)
    seed_everything(int(config['seed'][1]), workers=True)
    path_root = '../../'
    config['paths']['path_root'] = path_root

    # Ideal focal length
    ideal_focal_length = 285.75

    # Focal length sweep
    focal_length_sweep = np.linspace(ideal_focal_length*0.95, ideal_focal_length*1.05, 20)
    focal_length_sweep = np.append(focal_length_sweep, ideal_focal_length)
    focal_length_sweep = np.sort(focal_length_sweep)

    torch.save(focal_length_sweep, 'focal_length_sweep.pt')


    # Initialize the data
    #path_data = os.path.join(path_root, 'data/subsampled_data_3perclass.pt')
    #logger.info("Loading data from {}".format(path_data))
    #data = torch.load(path_data, weights_only=True)

    datamodule = datamodule.select_data(config)
    datamodule.setup()
    data = datamodule.val_dataloader()

    total_image_metrics = []
    total_feature_metrics = []
    total_classifier_metrics = []
    example_images = []

    model = CooperativeOpticalModelRemoteSim2Real.load_from_checkpoint('../../results/coop_sim2real_MNIST_bench_image/version_0/checkpoints/model-epoch=13.ckpt', params=config)
    model.cuda()


    # Run the datamodule
    with torch.no_grad():
        for j, d in enumerate(tqdm(data, desc='Running data', leave=False)):
            try:
                sample, slm_sample, target = d
                sample = sample.cuda()
                slm_sample = slm_sample.cuda()
                target = target.cuda()
                train_batch = (sample, slm_sample, target)
                # Shared step forward
                outputs = model.shared_step(train_batch)
                if model.crop_normalize_flag:
                    outputs['samples'] = model.crop_normalize(outputs['samples'])
                sample = outputs['samples'].squeeze().detach().cpu()
                simulation_image = outputs['simulation_image'].squeeze().detach().cpu()
                bench_image = outputs['bench_image'].squeeze().detach().cpu() 
                phases = outputs['phases'].squeeze()

                classifier_features = outputs['classifier_features']
                classifier_target = outputs['classifier_target'] # one-hot
                classifier_output = outputs['classifier_output'] # logits

                image_metrics = calculate_image_metrics([sample.abs(), bench_image, simulation_image])
                feature_metrics = calculate_feature_metrics(classifier_features)
                classifier_metrics = calculate_classifier_metrics(classifier_output, classifier_target)

                total_image_metrics.append(image_metrics)
                total_feature_metrics.append(feature_metrics)
                total_classifier_metrics.append(classifier_metrics)

                if j == 0:
                    example_images.append({
                        'sample': sample,
                        'bench_image': bench_image,
                        'simulation_image': simulation_image
                        })

            except Exception as e:
                logger.error(e)
                break

    # Cleanup
    model.upload_benign_image(which=0)
    model.upload_benign_image(which=1)

    output = {
            'image_metrics': total_image_metrics,
            'feature_metrics': total_feature_metrics,
            'classifier_metrics': total_classifier_metrics,
            'example_images': example_images
            }

    torch.save(output, 'Sim2Real_eval_val.pt')
    from IPython import embed; embed()

