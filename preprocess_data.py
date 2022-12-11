import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pip._internal.utils import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

label_to_index = {
    "red circle": 0,
    "red square": 1,
    "red triangle": 2,
    "red star": 3,
    "yellow circle": 4,
    "yellow square": 5,
    "yellow triangle": 6,
    "yellow star": 7,
    "green circle": 8,
    "green square": 9,
    "green triangle": 10,
    "green star": 11,
    "blue circle": 12,
    "blue square": 13,
    "blue triangle": 14,
    "blue star": 15,
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]

        return input, label

def get_image_tensors_array():
    labels = pd.read_csv("labels.csv")
    X = []
    y = []
    directory = "images"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            image = Image.open(f)
            # transform = transforms.Compose([
            #     transforms.PILToTensor()
            # ])
            # img_tensor = transform(image)
            transform = transforms.ToTensor()
            # Convert the image to PyTorch tensor
            img_tensor = transform(image)
            X.append(img_tensor)
            tag = str(labels[f][0])
            tags = tag.split("'")
            y.append([tags[1], tags[3]])

    return X, y

def train_test_split_12class_4class():
    X, y = get_image_tensors_array()
    class4_idx = []
    class12_idx = []
    for index, image in enumerate(X):
        if y[index][0] in ["green", "yellow"] and y[index][1] in ["triangle", "star"]:
            class4_idx.append(index)
        else:
            class12_idx.append(index)

    # TODO
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(class12_idx)):
        X_train.append(X[class12_idx[i]])
        y_train.append(label_to_index[y[class12_idx[i]][0] + " " + y[class12_idx[i]][1]])

    for i in range(len(class4_idx)):
        X_test.append(X[class4_idx[i]])
        y_test.append(label_to_index[y[class4_idx[i]][0] + " " + y[class4_idx[i]][1]])

    X_train = torch.stack(X_train)
    X_test = torch.stack(X_test)
    return X_train, y_train, X_test, y_test

def train_test_split_basic_classifier():
    X, y = get_image_tensors_array()
    dataset = Dataset(X, y)
    training_set, validation_set, test_set = torch.utils.random_split(dataset, [.7, .1, .2], generator=torch.Generator().manual_seed(42))
