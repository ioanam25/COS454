import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pip._internal.utils import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd

class Preprocessing():
    def get_image_tensors_array(self):
        labels = pd.read_csv("labels.csv")
        X = []
        y = []
        directory = "images"
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                image = Image.open(f)
                transform = transforms.Compose([
                    transforms.PILToTensor()
                ])
                img_tensor = transform(image)
                X.append(img_tensor)
                y.append(labels[f])

        return X, y

    def train_test_split(self, X, y):
        train_idx = []
        test_idx = []
        for index, image in enumerate(X):
            if y[index][0] in ["green", "yellow"] and y[index][1] in ["triangle", "star"]:
                test_idx.append(index)
            else:
                train_idx.append(index)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # N: number of examples in batch
        # C: number of channels
        H = 224 # H: image height in pixels
        W = 224 # W: image width in pixels
        K_c = 4 # K_c: number of colors
        K_s = 4 # K_s: number of shapes
        # Let x be image batch: tensor of shape [N, C, H, W]
        # Let encoder be an encoding function with final nonlinearity
        # input must be 224 x 224 for ResNet-18
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # Set up network layers
        self.D_pre = 1000 # output of resnet
        self.color_enc = nn.Linear(self.D_pre, K_c)
        self.shape_enc = nn.Linear(self.D_pre, K_s)

    # Forward pass
    def forward(self, x):
        z_pre = self.encoder(x)  # shape: [N, D_pre]
        z_c = self.color_enc(z_pre)  # shape: [N, K_c]
        z_s = self.shape_enc(z_pre)  # shape: [N, K_s]
        z_cs = torch.reshape(torch.unsqueeze(z_c, -1) + torch.unsqueeze(z_s, 1), (N, -1))  # shape: [N, K_c * K_s]
        return z_cs


model = Net()


def train_one_epoch(epoch_index, tb_writer, training_loader, loss_fn, optimizer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


class Train():
    def training_loop(self, training_set):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = 5
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        best_vloss = 1_000_000

        training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))
            model.train(True)
            avg_loss = train_one_epoch(self, epoch_number, writer, training_loader, loss_fn, optimizer)
            epoch_number += 1

if __name__ == '__main__':
    print_hi('PyCharm')


