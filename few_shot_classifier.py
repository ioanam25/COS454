import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
# from pip._internal.utils import datetime
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import preprocess_data


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()
        self.N = 1      # N: number of examples in batch
        self.C = 3       # C: number of channels
        self.H = 224     # H: image height in pixels
        self.W = 224     # W: image width in pixels
        self.K_c = 4     # K_c: number of colors
        self.K_s = 4     # K_s: number of shapes
        # Let x be image batch: tensor of shape [N, C, H, W]
        # Let encoder be an encoding function with final nonlinearity
        # input must be 224 x 224 for ResNet-18
        self.encoder = torch.hub.load('pytorch/vision:v0.2.2', 'resnet18', pretrained=False)
        # Set up network layers
        self.D_pre = 1000 # output of resnet
        self.cs_enc = nn.Linear(self.D_pre, 12)

    # Forward pass
    def forward(self, x):
        z_pre = self.encoder(x)  # shape: [N, D_pre]
        z_cs = self.cs_enc(z_pre)
        return z_cs


class Net4(nn.Module)


def train_one_epoch(epoch_index, tb_writer, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i <= 10 or i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def training():
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 1
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, optimizer, loss_fn)

        # We don't need gradients on to do reporting
        model.train(False)

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training Loss',
                           {'Training': avg_loss},
                           epoch_number + 1)
        writer.flush()

        epoch_number += 1

def testing():
    correct_pred = 0
    for i, data in enumerate(test_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        m = nn.Softmax(dim=1)
        output = m(model(inputs))
        print(torch.argmax(output[0]), labels[0])
        if torch.argmax(output[0]) == labels[0]:
            correct_pred += 1

    print(correct_pred, len(test_loader))
    return correct_pred / len(test_loader)


if __name__ == '__main__':
    model = ZeroShotNet()
    _, _, _, training_set, test_set = preprocess_data.train_test_split_12class_4class()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = nn.DataParallel(model)
    # torch.cuda.set_device(device)
    model.to(device)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set)

    training()

    print(testing())

    PATH = "trained_model_basic_classifier.pt"
    torch.save(model.state_dict(), PATH)
