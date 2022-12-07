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

    def train_test_split(self):
        X, y = self.get_image_tensors_array()
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
        for i in range(train_idx):
            X_train.append(X[train_idx[i]])
            y_train.append(y[train_idx[i]])

        for i in range(test_idx):
            X_test.append(X[test_idx[i]])
            y_test.append(y[test_idx[i]])

        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input = self.input[idx]
        label = self.labels[idx]
        sample = {"Image": input, "Labels": label}
        return sample


dataset = Preprocessing.train_test_split()
training_set = dataset[0], dataset[1]
test_set = dataset[2], dataset[3]
image_labels_df_train = pd.DataFrame({'Image': training_set[0], 'Labels': training_set[1]})
image_labels_df_test = pd.DataFrame({'Image': test_set[0], 'Labels': test_set[1]})

# define data set object
training_data = CustomDataset(image_labels_df_train['Image'], image_labels_df_train['Labels'])
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


class Train():
    def training_loop(self, training_set):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        # TODO EPOCHS = ???
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        training_loader = torch.utils.data.DataLoader(training_data, batch_size=20, shuffle=True, num_workers=2) # TODO choose batch size

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))
            running_loss = 0.

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
                    last_loss = running_loss / 1000  # loss per batch TODO check??
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    x = epoch * len(training_set) + i + 1
                    writer.add_scalar('Loss/train', last_loss, x)
                    running_loss = 0.


if __name__ == '__main__':
    print_hi('PyCharm')


