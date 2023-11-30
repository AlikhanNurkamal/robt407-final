import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from config import config
from models.cnn import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.utils import get_test_dataloader


def inference(test_loader, model, model_name, config):
    model.load_state_dict(torch.load(os.path.join(config['MODELS_DIR'], f'{model_name}_best_model.pth')))
    model.eval()

    df = pd.DataFrame()
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Inference'):
            img_names, imgs = data
            imgs = imgs.to(config['DEVICE'])

            logits = model(imgs)
            probas = F.softmax(logits, dim=1)

            block = np.concatenate((np.array(img_names).reshape(-1, 1), probas.detach().cpu().numpy()), axis=1)
            block = pd.DataFrame(block, columns=['img', 'c0', 'c1', 'c2',
                                                 'c3', 'c4', 'c5',
                                                 'c6', 'c7', 'c8',
                                                 'c9'])
            df = pd.concat((df, block), axis=0, ignore_index=True)
    
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory of test images.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name.')
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = args.images_dir
    model_name = args.model_name

    if 'resnet50' in model_name:
        model = ResNet50()
        model_name = 'resnet50'
    elif 'resnet101' in model_name:
        model = ResNet101()
        model_name = 'resnet101'
    elif 'custom' in model_name:
        model = customCNN()
        model_name = 'custom'
    else:
        raise NotImplementedError('unknown architecture')

    model = model.to(config['DEVICE'])
    config['TEST_DIR'] = images_dir
    test_loader = get_test_dataloader()
    
    # this csv file will be submitted to kaggle
    result = inference(test_loader, model, model_name, config)
    result.to_csv('result.csv', index=False)


if __name__ == '__main__':
    main()
