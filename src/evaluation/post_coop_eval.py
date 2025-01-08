
import os
import numpy as np
import torch
import yaml
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.append('../')
from models.models import CooperativeOpticalModelRemote
from datamodule import datamodule
from pytorch_lightning import seed_everything



if __name__ == "__main__":

    path_results = '../../results/coop_MNIST_bench_image/version_0'
    config = yaml.load(open(os.path.join(path_results, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    seed_everything(int(config['seed'][1]), workers=True)

    # Initialize the datamodule
    dm = datamodule.select_data(config)
    dm.setup()
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()

    path_checkpoint = os.path.join(path_results, 'checkpoints', 'last.ckpt')
    # For each focal length
    model = CooperativeOpticalModelRemote.load_from_checkpoint(path_checkpoint, params=config)
    model.cuda()

    train_preds = []
    train_targets = []

    # Run the datamodule
    with torch.no_grad():
        for train_batch in tqdm(train_loader, desc='Train Batch', leave=False):
            try:
                samples, slm_samples, classifier_targets = train_batch
                samples = samples.cuda()
                slm_samples = slm_samples.cuda()
                classifier_targets = classifier_targets.cuda()
                train_batch = (samples, slm_samples, classifier_targets)

                # Shared step forward
                outputs = model.shared_step(train_batch)
                classifier_target = outputs['classifier_target'] # one-hot
                classifier_output = outputs['classifier_output'] # logits

                # Convert to class index
                target_index = torch.argmax(classifier_target, dim=1).cpu().squeeze().numpy()
                pred_index = torch.argmax(classifier_output, dim=1).cpu().squeeze().numpy()

                train_preds.append(classifier_output)
                train_targets.append(classifier_target)
            except Exception as e:
                print(e)
                break

    # Compute the train f1 score
    #train_f1 = f1_score(train_targets, train_preds, average='macro')
    #train_f1_scores.append(train_f1)
    torch.save(train_preds, f'post_train_preds.pt')
    torch.save(train_targets, f'post_train_targets.pt')

    valid_preds = []
    valid_targets = []
    with torch.no_grad():
        for j,valid_batch in enumerate(tqdm(valid_loader, desc='Valid batch', leave=False)):
            try:
                samples, slm_samples, classifier_targets = valid_batch
                samples = samples.cuda()
                slm_samples = slm_samples.cuda()
                classifier_targets = classifier_targets.cuda()
                train_batch = (samples, slm_samples, classifier_targets)

                # Shared step forward
                outputs = model.shared_step(train_batch)
                classifier_target = outputs['classifier_target'] # one-hot
                classifier_output = outputs['classifier_output'] # logits

                # Convert to class index
                target_index = torch.argmax(classifier_target, dim=1).cpu().squeeze().numpy()
                pred_index = torch.argmax(classifier_output, dim=1).cpu().squeeze().numpy()

                valid_preds.append(classifier_output)
                valid_targets.append(classifier_target)
            except Exception as e:
                print(e)
                break
    
    # Compute the valid f1 score
    #valid_f1 = f1_score(valid_targets, valid_preds, average='macro')
    #valid_f1_scores.append(valid_f1)
    torch.save(valid_preds, f'post_valid_preds.pt')
    torch.save(valid_targets, f'post_valid_targets.pt')

    model.upload_benign_image(which=0)
    model.upload_benign_image(which=1)
