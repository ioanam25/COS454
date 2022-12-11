import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
#from pip._internal.utils import datetime
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

class BasicNet(nn.Module):
    def conv_block(self, in_size, out_size):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding = 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def __init__(self):
        super(BasicNet, self).__init__()
        self.C = 3  # C: number of channels
        self.H = 224  # H: image height in pixels
        self.W = 224  # W: image width in pixels
        # Let x be image batch: tensor of shape [N, C, H, W]
        # Let encoder be an encoding function with final nonlinearity
        # input must be 224 x 224 for ResNet-18
        
        #self.encoder = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=False)
        self.encoder = nn.Sequential(
            #self.conv_block(224*224*3, 64*64*3),
            #self.conv_block(64*64*3, 64*64*3),
            #self.conv_block(64*64*3, 64*64*3),
            #self.conv_block(64*64*3, 4096),
            self.conv_block(3, 3),
            Flatten()
        )
        self.linear = nn.Linear(37632, 16)
        #self.linear = nn.Linear(1000, 16)

    # Forward pass
    def forward(self, x):
        res_out = self.encoder(x)
        out = self.linear(res_out)
        return out


def train_one_epoch(epoch_index, tb_writer, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

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
        if i<=10 or i % 10 == 9:
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
    loss_fn = torch.nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, optimizer, loss_fn)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.cuda(non_blocking=True)
            vlabels = vlabels.cuda(non_blocking=True)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == '__main__':
    model = BasicNet()
    training_set, validation_set, test_set = preprocess_data.train_test_split_basic_classifier()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = nn.DataParallel(model)
    torch.cuda.set_device(device)
    model.cuda(device)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=2, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=2, shuffle=True, num_workers=0)

    training()

    PATH = "trained_model_basic_classifier.pt"
    torch.save(model.state_dict(), PATH)
