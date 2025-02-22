#
import os
import sys
import yaml
import torch
from tqdm import tqdm
sys.path.append('../')
from models.models import CooperativeOpticalModelRemote, Sim2Real
from utils.spatial_resample import spatial_resample
from datamodule import select_data
import pytorch_lightning as pl
import numpy as np


if __name__ == "__main__":
    pl.seed_everything(123)
    sys.path.append('../')
    params = yaml.load(open('../../config.yaml', 'r'), Loader=yaml.FullLoader)
    params['paths']['path_root'] = '../../'
    params['which'] = 'MNIST'
    params['paths']['path_data'] = 'data/'
    model = CooperativeOpticalModelRemote(params).cuda()

    path_data_baseline = os.path.join(params['paths']['path_root'], 'data', 'flat_field')
    os.makedirs(path_data_baseline, exist_ok=True)

    slm_sample = torch.zeros((1,1,1080, 1920))
    # Mask the central 1080,1080 region
    #y_center = 1080//2
    #x_center = 1920//2
    #slm_sample[:, :, y_center-540:y_center+540, x_center-540:x_center+540] = 255

    bench_image, lens_phase = model.bench_forward(slm_sample)
    bench_image = bench_image.squeeze().cpu()
    from IPython import embed; embed()
    torch.save(bench_image, os.path.join(path_data_baseline, 'flat_field.pt'))
