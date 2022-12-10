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

    print(X_train[0].shape)
    print(len(X_train))
    print(len(X_test))
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



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.N = 1      # N: number of examples in batch
        self.C = 3       # C: number of channels
        self.H = 224     # H: image height in pixels
        self.W = 224     # W: image width in pixels
        self.K_c = 4     # K_c: number of colors
        self.K_s = 4     # K_s: number of shapes
        # Let x be image batch: tensor of shape [N, C, H, W]
        # Let encoder be an encoding function with final nonlinearity
        # input must be 224 x 224 for ResNet-18
        self.encoder = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=False)
        # Set up network layers
        self.D_pre = 1000 # output of resnet
        self.color_enc = nn.Linear(self.D_pre, self.K_c)
        self.shape_enc = nn.Linear(self.D_pre, self.K_s)

    # Forward pass
    def forward(self, x):
        z_pre = self.encoder(x)  # shape: [N, D_pre]
        # print(z_pre.shape)
        z_c = self.color_enc(z_pre)  # shape: [N, K_c]
        # print(z_c.shape)
        z_s = self.shape_enc(z_pre)  # shape: [N, K_s]
        # print(z_s.shape)
        z_cs = torch.reshape(torch.unsqueeze(z_c, -1) + torch.unsqueeze(z_s, 1), (self.N, -1))  # shape: [N, K_c * K_s]
        # print(z_cs.shape)
        return z_cs


def training_loop(training_set, model):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    # TODO
    EPOCHS = 1
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=20, shuffle=True, num_workers=0) # TODO choose batch size

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        running_loss = 0

        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            print(i)
            print(inputs.shape)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(loss)

            # Gather data and report
            running_loss += loss.item()
            if i % 20 == 19:
                last_loss = running_loss / 20  # loss per batch TODO check??
                print('  batch {} loss: {}'.format((i + 1) / 20, last_loss))
                # x = epoch * len(training_set) + i + 1
                # writer.add_scalar('Loss/train', last_loss, x)
                running_loss = 0.



if __name__ == '__main__':
    model = Net()
    dataset = train_test_split()
    training = dataset[0], dataset[1]  # X_train y_train
    test = dataset[2], dataset[3]  # X_test y_test
    training_set = Dataset(dataset[0], dataset[1])

    training_loop(training_set, model)

    PATH = "trained_model.pt"
    torch.save(model.state_dict(), PATH)


