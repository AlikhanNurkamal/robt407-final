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
    
    print('Training finished!')
    return TRAIN_HISTORY, VAL_HISTORY


def main():
    # args = argparser()
    pass
