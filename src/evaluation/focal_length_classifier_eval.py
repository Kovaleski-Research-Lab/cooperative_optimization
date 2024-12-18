# Going to evaluate the classifier (trained on in-focus bench images) on 
# out-of-focus bench images. Going to sweep the focal length of the lens
# between +- 5% of the ideal focal length. At each focal length, we run the 
# dataset and save model predictions / ideal targets.
# The goal here is to find the 'line in the sand' on where the classifier
# starts to fail.

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

    config = yaml.load(open('../../config_coop.yaml', 'r'), Loader=yaml.FullLoader)
    seed_everything(int(config['seed'][1]), workers=True)
    path_root = '../../'
    config['paths']['path_root'] = path_root

    # Ideal focal length
    ideal_focal_length = 285.75

    # Focal length sweep
    focal_length_sweep = np.linspace(ideal_focal_length*0.95, ideal_focal_length*1.05, 20)
    focal_length_sweep = np.append(focal_length_sweep, ideal_focal_length)
    focal_length_sweep = np.sort(focal_length_sweep)

    train_f1_scores = []
    valid_f1_scores = []

    valid_images = []

    # Initialize the datamodule
    dm = datamodule.select_data(config)
    dm.setup()
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()

    # For each focal length
    for i,focal_length in enumerate(tqdm(focal_length_sweep, desc='Focal Length Sweep')):
        config['modulators'][1]['focal_length'] = focal_length
        model = CooperativeOpticalModelRemote(config)
        model.eval()
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

                    train_preds.append(pred_index)
                    train_targets.append(target_index)
                except Exception as e:
                    print(e)
                    break

        # Compute the train f1 score
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        train_f1_scores.append(train_f1)

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
                    if j == 0:
                        valid_images.append(outputs['bench_image'])
                    classifier_target = outputs['classifier_target'] # one-hot
                    classifier_output = outputs['classifier_output'] # logits

                    # Convert to class index
                    target_index = torch.argmax(classifier_target, dim=1).cpu().squeeze().numpy()
                    pred_index = torch.argmax(classifier_output, dim=1).cpu().squeeze().numpy()

                    valid_preds.append(pred_index)
                    valid_targets.append(target_index)
                except Exception as e:
                    print(e)
                    break
        
        # Compute the valid f1 score
        valid_f1 = f1_score(valid_targets, valid_preds, average='macro')
        valid_f1_scores.append(valid_f1)

    model.upload_benign_image(which=0)
    model.upload_benign_image(which=1)

    # Save the results
    results = {
        'focal_length_sweep': focal_length_sweep,
        'train_f1_scores': train_f1_scores,
        'valid_f1_scores': valid_f1_scores,
        'valid_images': valid_images
    }
    torch.save(results, 'focal_length_classifier_eval_results.pt')

