import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from pip._internal.utils import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from main import Net

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

index_to_label = {}
for key, value in label_to_index.items():
    index_to_label[value] = key


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


def train_test_split():
    X, y = get_image_tensors_array()
    train_idx = []
    test_idx = []
    for index, image in enumerate(X):
        if y[index][0] in ["green", "yellow"] and y[index][1] in ["triangle", "star"]:
            test_idx.append(index)
        else:
            train_idx.append(index)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(train_idx)):
        X_train.append(X[train_idx[i]])
        y_train.append(label_to_index[y[train_idx[i]][0] + " " + y[train_idx[i]][1]])

    for i in range(len(test_idx)):
        X_test.append(X[test_idx[i]])
        y_test.append(label_to_index[y[test_idx[i]][0] + " " + y[test_idx[i]][1]])

    print(torch.stack(X_train).shape)
    X_train = torch.stack(X_train)
    X_test = torch.stack(X_test)
    return X_train, y_train, X_test, y_test


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


if __name__ == '__main__':
    dataset = train_test_split()
    training = dataset[0], dataset[1]  # X_train y_train
    test = dataset[2], dataset[3]  # X_test y_test
    test_set = Dataset(dataset[2], dataset[3])
    test_loader = torch.utils.data.DataLoader(test_set)

    PATH = "trained_model.pt"
    model = Net()
    model.load_state_dict(torch.load(PATH))

    print("model loaded")

    count_success = 0
    count_total = 0
    for inputs, labels in test_set:
        inputs = inputs[None, :]
        y_pred = model(inputs)
        # print(y_pred)
        ans = F.softmax(y_pred, dim=1)
        # print(ans[0])
        ans_no = np.argmax(ans[0].detach().numpy())
        # print(ans_no)
        # print(index_to_label[ans_no], index_to_label[labels])
        print(ans_no, labels)
        count_total += 1
        count_success += (ans_no == labels)

    print(count_success, count_total)
    print(count_success / count_total)
