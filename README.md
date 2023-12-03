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

## Inference
There is also an inference code provided so that you can test pretrained models on your own dataset. First, download pretrained weights via this [link](https://drive.google.com/drive/folders/1Tcb2s1_tBHjh2VSAn2qgDPl_a0bnjBSk?usp=sharing), store them in the directory "model_weights", and run
```
python3 inference.py --images_dir dir_to_test_images --model_name your_desired_model
```
The directory of test images should look like this:
```
Images
├── img1.jpg
├── img2.jpg
├── img3.jpg
├── ...
```

## Results
You can see our results (Kaggle competition scores) below:
- CVT
![alt text](https://drive.google.com/file/d/1R3ZmrjUE3Nlq1FE9aKg3jPWUmzNsJBhe/view?usp=share_link "Cvt results")
- ViT lite
![alt text](https://drive.google.com/file/d/1BbQXw20cGIOn1fl0J0DTmweziigaOdFd/view?usp=share_link "ViT Lite")
- Resnet
![alt text](https://drive.google.com/file/d/1Vzf9W-tlJRn5dZdaR9gNQ397mJQAgdgX/view?usp=share_link "Resnet")

