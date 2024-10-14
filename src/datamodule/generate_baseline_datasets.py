import os
import sys
import yaml
import torch
from tqdm import tqdm
sys.path.append('../')
from models.models import CooperativeOpticalModel
from utils.spatial_resample import spatial_resample
from datamodule import select_data
import pytorch_lightning as pl


if __name__ == "__main__":
    pl.seed_everything(123)
    sys.path.append('../')
    params = yaml.load(open('../../config.yaml', 'r'), Loader=yaml.FullLoader)
    params['paths']['path_root'] = '../../'
    model = CooperativeOpticalModel(params).cuda()

    datamodule = select_data(params)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.val_dataloader()
    scaled_plane = model.dom.layers[0].input_plane.scale(0.53, inplace=False)


    path_data_baseline = os.path.join(params['paths']['path_root'], 'data', 'baseline')
    os.makedirs(path_data_baseline, exist_ok=True)

    for i,batch in enumerate(tqdm(train_dataloader)):
        sample, slm_sample, target = batch
        sim_output = model.dom_forward(sample.cuda())
        bench_image, lens_phase = model.bench_forward(slm_sample)
        resampled_sample = spatial_resample(scaled_plane, sample.abs(), model.dom.layers[1].output_plane).squeeze()
        bench_image = bench_image.squeeze().abs().cpu()
        sim_output = sim_output.squeeze().abs().cpu()**2
        new_images = {'resampled_sample': resampled_sample, 
                      'bench_image': bench_image, 
                      'sim_output': sim_output,
                      'target': target}
                
        torch.save(new_images, os.path.join(path_data_baseline, f'baseline_train_{i:04d}.pt'))

    for i,batch in enumerate(tqdm(valid_dataloader)):
        sample, slm_sample, target = batch
        sim_output = model.dom_forward(sample.cuda())
        bench_image, lens_phase = model.bench_forward(slm_sample)
        resampled_sample = spatial_resample(scaled_plane, sample.abs(), model.dom.layers[1].output_plane).squeeze()
        bench_image = bench_image.squeeze().abs().cpu()
        sim_output = sim_output.squeeze().abs().cpu()**2
        new_images = {'resampled_sample': resampled_sample, 
                      'bench_image': bench_image, 
                      'sim_output': sim_output,
                      'target': target}
                
        torch.save(new_images, os.path.join(path_data_baseline, f'baseline_valid_{i:04d}.pt'))
