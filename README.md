# ROBT407 Final Project
This repository contains the final project for the course ROBT407 - Machine Learning with Applications.

## Models
In this project, we implemented
1. ResNet (50 and 101) architecture as well as custom CNN model (VGG16-like).
2. ViT-lite as well as CVT (Compact Convolutional Transformer) to detect distracted drivers from images.

## Training
In order to train a model, run the following script
```
python3 train.py --model_name your_desired_model
```
In case you want to train a resnet50 or resnet101, run `--model_name resnet50` or `--model_name resnet101`. In case you want to train a custom CNN model (which is a lot like VGG16), run `--model_name custom`

## Training Results
In the “Training Results” directory, each model’s subfolder contains its training code, results, accuracy, F1, and loss graphs.
```
Trainnig Results
├── Model name
│   ├── Accuracy graph
│   ├── F1 Score graph
│   ├── Loss graph
│   ├── submission.csv
│   └── Training code
```
