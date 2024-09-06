import os
import yaml
import sys
import torch
from loguru import logger
from IPython import embed
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


from src import datamodule
from src import models


def get_next_version(path_results,name):
    version = 0
    while os.path.exists(os.path.join(path_results, name, 'version_{}'.format(version))):
        version += 1
    return version

def run(params):
    logger.info('Running the model')

    # Initialize seeding
    if params['seed'][0]:
        seed_everything(params['seed'][1], workers=True)

    alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']

    # Initialize the paths
    path_root = os.getcwd()
    params['paths']['path_root'] = path_root
    path_results = params['paths']['path_results']
    path_data = params['paths']['path_data']
    path_checkpoints = params['paths']['path_checkpoints']
    path_images = params['paths']['path_images']

    path_results = os.path.join(path_root, path_results)
    path_data = os.path.join(path_root, path_data)
    name = 'alpha_{}_beta_{}_gamma_{}'.format(alpha, beta, gamma)
    version = get_next_version(path_results, name)
    path_images = os.path.join(path_root, path_results, name, f'version_{version}', path_images)
    params['paths']['path_images'] = path_images

    # Initialize the CSV logger
    save_dir = os.path.join(path_results, name)
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
            )

    # Initialize the model
    model = models.CooperativeOpticalModel(params)

    # Initialize the datamodel
    dm = datamodule.select_data(params)

    # Get the training params
    gpu_list = params['gpu_config'][1]
    num_epochs = params['num_epochs']

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
                num_sanity_val_steps = 0,
                check_val_every_n_epoch = num_epochs +1,
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
                num_sanity_val_steps = 0,
                check_val_every_n_epoch = num_epochs +1,
                callbacks = [model_checkpoint],
                logger = csv_logger,
                )

    # Fit the model
    trainer.fit(model, dm)

    # Dump the config
    yaml.dump(params, open(os.path.join(path_root, path_results,name,f'version_{version}', 'config.yaml'), 'w'))

def get_alpha_beta_gamma_from_cla(argv):
    if len(argv) < 4:
        logger.error('Usage: python train.py alpha beta gamma')
        sys.exit(1)
    alpha = float(argv[1])
    beta = float(argv[2])
    gamma = float(argv[3])
    return alpha, beta, gamma

if __name__ == "__main__":
    # Load the parameters
    params = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    alpha, beta, gamma = get_alpha_beta_gamma_from_cla(sys.argv)
    params['alpha'] = alpha
    params['beta'] = beta
    params['gamma'] = gamma

    params['paths']['path_root'] = os.getcwd()

    run(params)

