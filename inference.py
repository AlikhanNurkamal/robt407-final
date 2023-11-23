import os
import pandas as pd
import argparse
from tqdm import tqdm
from config import config
from models.cnn import *

import torch
import torch.nn as nn
import utils.utils as utils
from utils.utils import CNNInferenceDataset
from torch.utils.data import DataLoader


def cnn_inference(test_loader, model, model_name, config):
    if model_name == 'resnet50' or model_name == 'resnet101':
        model.fc = nn.Linear(2048, 10)
    else:
        raise NotImplementedError('unknown architecture')
    
    model = model.to(config['DEVICE'])
    model.load_state_dict(torch.load(os.path.join(config['MODELS_DIR'], f'{model_name}_best_model.pth')))
    model.eval()

    img_names = []
    predictions = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Inference'):
            img_name, imgs = data
            imgs = imgs.to(config['DEVICE'])

            logits = model(imgs)
            img_names.extend(img_name)
            predictions.extend(logits.detach().cpu().numpy())
    
    # convert predictions to columns to submit to kaggle
    columns = pd.get_dummies(predictions)
    res = pd.DataFrame({'img': img_names})
    res = pd.concat([res, columns], axis=1)
    res.rename(columns={0: 'c0', 1: 'c1', 2: 'c2',
                        3: 'c3', 4: 'c4', 5: 'c5',
                        6: 'c6', 7: 'c7', 8: 'c8',
                        9: 'c9'}, inplace=True)
    res.replace({False: 0, True: 1}, inplace=True)

    return res


def rnn_inference():
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True,
                        help='Task number. 1 for Distracted Driver Detection, 2 for Quora Insincere Questions Classification')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task
    model_name = args.model

    if task == 1:
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

        test_transformations = utils.val_transforms
        test_dataset = CNNInferenceDataset(config['TEST_DIR'], transform=test_transformations)
        test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'])

        # this csv file will be submitted to kaggle
        result = cnn_inference(test_loader, model, model_name, config)
        return result
    elif task == 2:
        pass
    else:
        raise Exception('unknown task')


if __name__ == '__main__':
    main()
