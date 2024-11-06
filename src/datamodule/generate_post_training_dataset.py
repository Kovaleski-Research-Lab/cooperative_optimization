import os
import sys
import yaml
import torch
from tqdm import tqdm
sys.path.append('../')
from models.models import CooperativeOpticalModelRemote
from utils.spatial_resample import spatial_resample
from datamodule import select_data
import pytorch_lightning as pl


if __name__ == "__main__":
    pl.seed_everything(123)
    sys.path.append('../')
    params = yaml.load(open('../../config.yaml', 'r'), Loader=yaml.FullLoader)
    params['paths']['path_root'] = '../../'
    path_checkpoint = '../../results/coop_bench_alpha_0.0_beta_0.0_gamma_0.0_delta_1.0/version_4/checkpoints/model-epoch=13.ckpt'
    config = yaml.load(open('../../results/coop_bench_alpha_0.0_beta_0.0_gamma_0.0_delta_1.0/version_4/config.yaml', 'r'), Loader=yaml.FullLoader)
    checkpoint = torch.load(path_checkpoint, weights_only=True)
    model = CooperativeOpticalModelRemote(params).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    datamodule = select_data(config)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.val_dataloader()
    scaled_plane = model.dom.layers[0].input_plane.scale(0.53, inplace=False)


    path_data_baseline = os.path.join(params['paths']['path_root'], 'data', 'post_training_v4')
    os.makedirs(path_data_baseline, exist_ok=True)

    model.eval()

    for i,batch in enumerate(tqdm(train_dataloader)):
        sample, slm_samples, target = batch
        batch = (sample.cuda(), slm_samples.cuda(), target.cuda())
        outputs = model.shared_step(batch)
        simulation_image = outputs['simulation_outputs']['images']
        bench_image = outputs['bench_image']
        phases = outputs['phases']
        classifier_output = outputs['classifier_output']
        classifier_target = outputs['classifier_target']

        resampled_sample = spatial_resample(scaled_plane, sample.abs(), model.dom.layers[1].output_plane).squeeze()
        bench_image = bench_image.squeeze().abs().cpu()
        sim_output = simulation_image.squeeze().cpu()

        new_images = {'resampled_sample': resampled_sample, 
                      'bench_image': bench_image, 
                      'sim_output': sim_output,
                      'target': target}
                
        torch.save(new_images, os.path.join(path_data_baseline, f'postTrain_train_{i:04d}.pt'))

    for i,batch in enumerate(tqdm(valid_dataloader)):
        sample, slm_samples, target = batch
        batch = (sample.cuda(), slm_samples.cuda(), target.cuda())
        outputs = model.shared_step(batch)
        outputs = model.shared_step(batch)
        simulation_image = outputs['simulation_outputs']['images']
        bench_image = outputs['bench_image']
        phases = outputs['phases']
        classifier_output = outputs['classifier_output']
        classifier_target = outputs['classifier_target']

        resampled_sample = spatial_resample(scaled_plane, sample.abs(), model.dom.layers[1].output_plane).squeeze()
        bench_image = bench_image.squeeze().abs().cpu()
        sim_output = simulation_image.squeeze().cpu()

        new_images = {'resampled_sample': resampled_sample, 
                      'bench_image': bench_image, 
                      'sim_output': sim_output,
                      'target': target}
                
        torch.save(new_images, os.path.join(path_data_baseline, f'postTrain_valid_{i:04d}.pt'))
