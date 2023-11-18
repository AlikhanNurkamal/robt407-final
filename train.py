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


def train(train_loader, model, loss_fn, optimizer, device):
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
    return avg_loss, accuracy, precision, recall, f1


def argparser():
    pass


def main():
    args = argparser()
