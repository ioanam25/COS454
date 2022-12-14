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
import torch.utils

label_to_index = {
    "red circle": 0,
    "red square": 1,
    "red triangle": 2,
    "red star": 3,
    "yellow circle": 4,
    "yellow square": 5,
    "green circle": 6,
    "green square": 7,
    "blue circle": 8,
    "blue square": 9,
    "blue triangle": 10,
    "blue star": 11,

    "yellow triangle": 12,
    "yellow star": 13,
    "green triangle": 14,
    "green star": 15,
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
            transform = transforms.ToTensor()
            # Convert the image to PyTorch tensor
            img_tensor = transform(image)
            X.append(img_tensor)
            tag = str(labels[f][0])
            tags = tag.split("'")
            y.append([tags[1], tags[3]])

    return X, y


def train_test_split_12class_4class_zero_shot():
    X, y = get_image_tensors_array()
    class4_idx = []
    class12_idx = []
    for index, image in enumerate(X):
        if y[index][0] in ["green", "yellow"] and y[index][1] in ["triangle", "star"]:
            class4_idx.append(index)
        else:
            class12_idx.append(index)

    X_class12 = []
    y_class12 = []
    X_class4 = []
    y_class4 = []
    for i in range(len(class12_idx)):
        X_class12.append(X[class12_idx[i]])
        y_class12.append(label_to_index[y[class12_idx[i]][0] + " " + y[class12_idx[i]][1]])

    for i in range(len(class4_idx)):
        X_class4.append(X[class4_idx[i]])
        y_class4.append(label_to_index[y[class4_idx[i]][0] + " " + y[class4_idx[i]][1]])

    dataset12 = Dataset(X_class12, y_class12)
    training_set12, validation_set12, test_set12 = torch.utils.data.random_split(dataset12, [.7, .1, .2],
                                                                                 generator=torch.Generator().manual_seed(
                                                                                     42))
    dataset4 = Dataset(X_class4, y_class4)
    training_set4, test_set4 = torch.utils.data.random_split(dataset4, [.01, .99],
                                                             generator=torch.Generator().manual_seed(42))

    return training_set12, validation_set12, test_set12, training_set4, test_set4


def train_test_split_12class_4class_few_shot():
    X, y = get_image_tensors_array()
    class4_idx = []
    class12_idx = []
    for index, image in enumerate(X):
        if y[index][0] in ["green", "yellow"] and y[index][1] in ["triangle", "star"]:
            class4_idx.append(index)
        else:
            class12_idx.append(index)

    X_class12 = []
    y_class12 = []
    X_class4 = []
    y_class4 = []
    for i in range(len(class12_idx)):
        X_class12.append(X[class12_idx[i]])
        y_class12.append(label_to_index[y[class12_idx[i]][0] + " " + y[class12_idx[i]][1]])

    for i in range(len(class4_idx)):
        X_class4.append(X[class4_idx[i]])
        y_class4.append(label_to_index[y[class4_idx[i]][0] + " " + y[class4_idx[i]][1]] - 12)

    dataset12 = Dataset(X_class12, y_class12)
    training_set12, validation_set12, test_set12 = torch.utils.data.random_split(dataset12, [.7, .1, .2],
                                                                                 generator=torch.Generator().manual_seed(
                                                                                     42))
    dataset4 = Dataset(X_class4, y_class4)
    training_set4, test_set4 = torch.utils.data.random_split(dataset4, [.01, .99],
                                                             generator=torch.Generator().manual_seed(42))

    return training_set12, validation_set12, test_set12, training_set4, test_set4


def train_test_split_basic_classifier():
    X, y = get_image_tensors_array()
    yy = []
    for i in range(len(y)):
        yy.append(label_to_index[y[i][0] + " " + y[i][1]])
    dataset = Dataset(X, yy)
    training_set, validation_set, test_set = torch.utils.data.random_split(dataset, [.7, .1, .2],
                                                                           generator=torch.Generator().manual_seed(42))
    return training_set, validation_set, test_set
