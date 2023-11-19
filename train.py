import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import ResNet50, ResNet101
from utils import CustomDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, precision, recall, f1


def train(train_loader, model, loss_fn, optimizer, epoch, device):
    model.train()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    for data in tqdm(train_loader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # get predictions as the index of max logit
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        total_loss += loss.item()
    
    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch} | Train Loss: {avg_loss} | F1: {f1} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}")
    return avg_loss, accuracy, precision, recall, f1


def validate(val_loader, model, loss_fn, device):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    for data in tqdm(val_loader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(imgs)
            loss = loss_fn(logits, labels)

        # get predictions as the index of max logit
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        total_loss += loss.item()
    
    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(val_loader)

    print(f"\tValidation Loss: {avg_loss} | F1: {f1} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}")
    return avg_loss, accuracy, precision, recall, f1


def train_and_validate(train_loader, val_loader, model, loss_fn, optimizer, config):# epochs=EPOCHS, patience=PATIENCE, device=DEVICE):
    TRAIN_HISTORY = {
        'Loss': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': []
    }
    VAL_HISTORY = {
        'Loss': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': []
    }

    # for early stopping
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(config['EPOCHS']):
        adjust_learning_rate(optimizer=optimizer, epoch=epoch)
        loss, accuracy, precision, recall, f1 = train(train_loader, model, loss_fn, optimizer, epoch + 1, config['DEVICE'])
        TRAIN_HISTORY['Loss'].append(loss)
        TRAIN_HISTORY['Accuracy'].append(accuracy)
        TRAIN_HISTORY['Precision'].append(precision)
        TRAIN_HISTORY['Recall'].append(recall)
        TRAIN_HISTORY['F1'].append(f1)

        loss, accuracy, precision, recall, f1 = validate(val_loader, model, loss_fn, config['DEVICE'])
        VAL_HISTORY['Loss'].append(loss)
        VAL_HISTORY['Accuracy'].append(accuracy)
        VAL_HISTORY['Precision'].append(precision)
        VAL_HISTORY['Recall'].append(recall)
        VAL_HISTORY['F1'].append(f1)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter == config['PATIENCE']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
    save_graphs(TRAIN_HISTORY['Loss'], VAL_HISTORY['Loss'], type="Loss")
    save_graphs(TRAIN_HISTORY['Accuracy'],VAL_HISTORY['Accuracy'], type='Accuracy')
    save_graphs(TRAIN_HISTORY['F1'],VAL_HISTORY['F1'], type='F1')
    
    print('Training finished!')
    return TRAIN_HISTORY, VAL_HISTORY


def adjust_learning_rate(optimizer, epoch, warmup=True, warmup_ep=10, enable_cos=True):
    lr = lr_init
    if warmup and epoch < warmup_ep:
        lr = lr / (warmup_ep - epoch)
    elif enable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_ep) / (total_epochs - warmup_ep)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def save_graphs(train, test, type="None"): 
    plt.figure(figsize=(10,5))
    plt.title(f"Training and Test {type}")
    plt.plot(test,label="test")
    plt.plot(train,label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{type}.png')


def main():
    
    img_resize = 224 # TODO add to config
    
    model = ResNet50()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(models.paramters(), lr=0.001, weight_decay=0.01)  # TODO add to config
    
    normalize = [] # [transforms.Normalize(mean=img_mean, std=img_std)]  # TODO add to config
    augmentatinos = []
    augmentatinos += [transforms.Resize(img_resize),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      *normalize]
    
    augmentatinos = transforms.Compose(augmentatinos)
    
    train_dataset = CustomDataset(images_dir='dir', transform=augmentatinos)
    # val_dataset = 
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # TODO add to config
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    train_and_validate(train_loader, val_loader, model, criterion, optimizer, config)
    
    
    # Memory consumtion, training time??