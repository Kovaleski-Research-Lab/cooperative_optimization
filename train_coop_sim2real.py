import os
import yaml
import sys
import torch
import argparse
from loguru import logger
from IPython import embed
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


from src.datamodule import datamodule
from src.models import models


def get_next_version(path_results,name):
    version = 0
    while os.path.exists(os.path.join(path_results, name, 'version_{}'.format(version))):
        version += 1
    return version

def get_current_version(path_results,name):
    version = 0
    while os.path.exists(os.path.join(path_results, name, 'version_{}'.format(version))):
        version += 1
    return version - 1

def run(params):
    logger.info('Running the model')

    # Initialize seeding
    if params['seed'][0]:
        seed_everything(params['seed'][1], workers=True)

    which = params['which']
    which_data = params['which_data']

    # Initialize the paths
    #path_root = '/develop'
    #params['paths']['path_root'] = path_root
    path_root = params['paths']['path_root']
    path_results = params['paths']['path_results']
    path_data = params['paths']['path_data']
    path_checkpoints = params['paths']['path_checkpoints']
    path_lens_phase = params['paths']['path_lens_phase']

    path_results = os.path.join(path_root, path_results)
    path_data = os.path.join(path_root, path_data)
    name = 'coop_sim2real_{}_{}'.format(which, which_data)

    if params['resume_training']:
        version = get_current_version(path_results, name)
    else:
        version = get_next_version(path_results, name)

    # Initialize the directory for saving lens phase
    params['paths']['path_lens_phase'] = os.path.join(path_results, name, f'version_{version}', path_lens_phase)
    lens_phase_dir = os.path.join(params['paths']['path_lens_phase'])
    os.makedirs(lens_phase_dir, exist_ok=True)

    # Initialize the CSV logger
    save_dir = os.path.join(path_results, name)
    print(save_dir)
    csv_logger = CSVLogger(save_dir = save_dir, version = 'logs', name = 'version_{}'.format(version))

    # Initialize the model checkpoint
    checkpoint_dir = os.path.join(path_results, name, f'version_{version}', path_checkpoints)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_checkpoint = ModelCheckpoint(
            dirpath = checkpoint_dir,
            filename = 'model-{epoch:02d}',
            save_last = True,
            verbose = False,
            save_on_train_epoch_end = True,
            monitor = 'loss_train',
            save_top_k = 2,
            )

    model = models.select_new_model(params)

    # Initialize the datamodel
    dm = datamodule.select_data(params)

    # Get the training params
    gpu_list = params['gpu_config'][1]
    num_epochs = params['num_epochs']

    # Dump the config
    yaml.dump(params, open(os.path.join(path_root, path_results,name,f'version_{version}', 'config.yaml'), 'w'))

    if params['resume_training']:
        resume_from_checkpoint = os.path.join(checkpoint_dir, 'last.ckpt')
    else:
        resume_from_checkpoint = None

    # Initialize the PyTorch Lightning Trainer
    if params['gpu_config'][0] and torch.cuda.is_available():
        logger.info('Using GPUs')
        trainer = Trainer(
                accelerator = 'cuda',
                num_nodes = 1,
                devices = gpu_list,
                max_epochs = num_epochs,
                deterministic = True,
                enable_progress_bar=True,
                enable_model_summary=True,
                log_every_n_steps = 1,
                default_root_dir = path_results,
                num_sanity_val_steps = 1,
                check_val_every_n_epoch = 1,
                callbacks = [model_checkpoint],
                logger = csv_logger,
                )
    else:
        logger.info('Using CPU')
        trainer = Trainer(
                accelerator = "cpu",
                max_epochs = num_epochs,
                deterministic = True,
                enable_progress_bar=True,
                enable_model_summary=True,
                log_every_n_steps = 1,
                default_root_dir = path_results,
                num_sanity_val_steps = 1,
                check_val_every_n_epoch = 1,
                callbacks = [model_checkpoint],
                logger = csv_logger,
                )

    # Fit the model
    trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    # Load the parameters
    params = yaml.load(open('config_coop_sim2real.yaml', 'r'), Loader=yaml.FullLoader)
    params['paths']['path_root'] = os.getcwd()

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--which_data", type=str, default='resampled_sample', help="Which element of the data to use")
    args = argparser.parse_args()
    params['which_data'] = args.which_data
    print("Running with the following parameters:")
    print(f"which_data: {params['which_data']}")
    run(params)

