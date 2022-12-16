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
        self.fc = nn.Linear(self.D_pre, 12)

    # Forward pass
    def forward(self, x):
        z_pre = self.encoder(x)  # shape: [N, D_pre]
        z_cs = self.fc(z_pre)
        return z_cs


def train_one_epoch(epoch_index, tb_writer, optimizer, loss_fn, training_loader):
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
        if i<=10 or i % 50 == 49:
            last_loss = running_loss / 10  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def training(training_loader):
    print("model:" + str(curr_model))
    # Initializing in a separate cell so we can easily add more epochs to the same run
    PATH12 = "models/few_model_12_{}_{}"
    PATH4 = "models/few_model_4_{}_{}"

    if curr_model == 12:
        PATH = PATH12
    else:
        PATH = PATH4

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    best_vloss = 1_000_000.
    no_improve = 0

    EPOCHS = 100
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if curr_model == 4:
        params = []
        for child in list(model.children())[:-1]:
            params.extend(list(child.parameters()))

        param_groups = [
            {'params': model.fc.parameters(), 'lr': .01},
            {'params': params, 'lr': .001},
        ]
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
        # optimizer = Adam(param_groups)

    best_path = ' '
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
        print('EPOCH {}:'.format(epoch + 1), file=f)

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer, optimizer, loss_fn, training_loader)

        # We don't need gradients on to do reporting
        model.train(False)

        if curr_model == 12:
            running_vloss = 0.0
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss), file=f)

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                no_improve = 0
                best_vloss = avg_vloss
                model_path = PATH.format(timestamp, epoch)
                best_path = model_path
                torch.save(model.state_dict(), model_path)
            else:
                no_improve += 1
                if no_improve > 8:
                    model_path = PATH.format(timestamp, epoch)
                    torch.save(model.state_dict(), model_path)
                    break
    return best_path

def testing():
    correct_pred = 0
    print(len(test_loader))
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
    f = open("few_output", "w")
    curr_model = 12
    model = Net12()
    training_set12, validation_set12, test_set12, training_set4, test_set4 = preprocess_data.train_test_split_12class_4class_few_shot()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    print(device, file=f)
    #model = nn.DataParallel(model)
    # torch.cuda.set_device(device)
    model.to(device)

    training_loader12 = torch.utils.data.DataLoader(training_set12, batch_size=1, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_set12, batch_size=1, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set12)

    best_path = training(training_loader12)
    # print(testing())

    # for param in model.parameters():
    #     param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
    model.load_state_dict(torch.load(best_path))
    model.fc = nn.Linear(1000, 4)  # assuming that the fc7 layer has 512 neurons, otherwise change it
    model.to(device)

    curr_model = 4
    training_loader4 = torch.utils.data.DataLoader(training_set4, batch_size=1, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set4)

    training(training_loader4)
    #print(testing())

