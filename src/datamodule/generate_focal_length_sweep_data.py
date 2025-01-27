import torch
import os
import sys
import pytorch_lightning as pl
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm
from pytorch_lightning import seed_everything
import torchvision

from datamodule import select_data
sys.path.append('../')
from models import models



if __name__ == "__main__":

    coop_config = yaml.load(open('../../config_coop.yaml', 'r'), Loader=yaml.FullLoader)
    sim2real_config = yaml.load(open('../../results/sim2real/version_0/config.yaml', 'r'), Loader=yaml.FullLoader)
    seed_everything(int(coop_config['seed'][1]), workers=True)
    path_root = '../../'
    coop_config['paths']['path_root'] = path_root
    sim2real_config['paths']['path_root'] = path_root

    # Ideal focal length
    ideal_focal_length = 285.75

    # Focal length sweep
    focal_length_sweep = np.linspace(ideal_focal_length*0.95, ideal_focal_length*1.05, 20)
    focal_length_sweep = np.append(focal_length_sweep, ideal_focal_length)
    focal_length_sweep = np.sort(focal_length_sweep)

    torch.save(focal_length_sweep, 'focal_length_sweep.pt')

    # Initialize the data
    path_data = os.path.join(path_root, 'data/subsampled_data_3perclass.pt')
    logger.info("Loading data from {}".format(path_data))
    data = torch.load(path_data, weights_only=True)

    total_image_metrics = []
    total_feature_metrics = []
    total_classifier_metrics = []
    example_images = []

    crop = torchvision.transforms.CenterCrop((1080,1080))

    output = {}
    # For each focal length
    for i,focal_length in enumerate(tqdm(focal_length_sweep, desc='Focal Length Sweep')):

        output[focal_length] = {}

        coop_config['modulators'][1]['focal_length'] = focal_length
        sim2real_config['modulators'][2]['focal_length'] = focal_length
        
        coop_model = models.CooperativeOpticalModelRemote(coop_config)
        coop_model.cuda()
        sim2real_model = models.Sim2Real.load_from_checkpoint('../../results/sim2real/version_0/checkpoints/last.ckpt', params=sim2real_config)
        sim2real_model.cuda()
        sim2real_model.dom.layers[2].modulator = coop_model.dom.layers[1].modulator

        sim2real_images = []
        coop_bench_images = []
        coop_sim_images = []
        # Run the datamodule
        with torch.no_grad():
            for j, d in enumerate(tqdm(data, desc='Running data', leave=False)):
                try:
                    sample, slm_sample, target = d
                    sample = sample.cuda()
                    slm_sample = slm_sample.cuda()
                    target = target.cuda()
                    train_batch = (sample, slm_sample, target)
                    coop_output = coop_model.shared_step(train_batch)
                    sim2real_output = sim2real_model.forward(sample)
                    sim2real_image = sim2real_output.squeeze().detach().cpu().abs()**2
                    if coop_model.crop_normalize_flag:
                        sim2real_image = coop_model.crop_normalize(sim2real_image)

                    sim2real_images.append(sim2real_image)
                    coop_bench_images.append(coop_output['bench_image'].squeeze().detach().cpu())
                    coop_sim_images.append(coop_output['simulation_outputs']['images'].squeeze().detach().cpu())

                except Exception as e:
                    logger.error(e)
                    break
        output[focal_length]['sim2real_images'] = sim2real_images
        output[focal_length]['coop_bench_images'] = coop_bench_images
        output[focal_length]['coop_sim_images'] = coop_sim_images

    # Cleanup
    coop_model.upload_benign_image(which=0)
    coop_model.upload_benign_image(which=1)


    torch.save(output, 'new_focal_length_sweep_data.pt')
    from IPython import embed; embed()




 
