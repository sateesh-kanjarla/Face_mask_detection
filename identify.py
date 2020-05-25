import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import time
import copy
from PIL import Image
import glob
import cv2

torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_path = 'weights.pth'
model = torch.load(file_path)


class_names = ['mask','no_mask']



def image_processing(image):
    pil_image = image
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = image_transforms(pil_image)
    return img
    
    


def label_identification(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image)
    image = image_processing(im)
    img = image.unsqueeze_(0)
    img = image.float()
    model.eval()
    model.cpu()
    output = model(image)
    _, predicted = torch.max(output, 1)
    classification1 = predicted.data[0]
    index = int(classification1)
    return class_names[index]












