import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vit_h_14
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import sys
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from pipeline_tool.dataset import PipelineDataset

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_input_shape(self):
        return [4, 3, 32, 32]
    
    def get_output_shape(self):
        return [4]
    
    def get_dtype(self):
        return "long"
    
    def get_trainloader(self):
        dataset = PipelineDataset(1024, self.get_input_shape()[1:], [1] if len(self.get_output_shape()) == 1 else self.get_output_shape()[1:], "long")
        return torch.utils.data.DataLoader(dataset, batch_size=self.get_input_shape()[0], shuffle=True)
    
class FFNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
    
    def get_input_shape(self):
        return [4, 28*28]
    
    def get_output_shape(self):
        return [4]
    
    def get_dtype(self):
        return "long"
    
    def get_trainloader(self):
        dataset = PipelineDataset(1024, self.get_input_shape()[1:], [1] if len(self.get_output_shape()) == 1 else self.get_output_shape()[1:], self.get_dtype())
        return torch.utils.data.DataLoader(dataset, batch_size=self.get_input_shape()[0], shuffle=True)

class Debug(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
    
    def get_input_shape(self):
        return [2, 4]
    
    def get_output_shape(self):
        return [2]
    
    def get_dtype(self):
        return "long"
    
    def get_trainloader(self):
        dataset = PipelineDataset(1024, self.get_input_shape()[1:], [1] if len(self.get_output_shape()) == 1 else self.get_output_shape()[1:], self.get_dtype())
        return torch.utils.data.DataLoader(dataset, batch_size=self.get_input_shape()[0], shuffle=True)

class BigModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = vit_h_14(weights='DEFAULT')
    
    def forward(self, image):
        return self.model(image)
    
    def get_input_shape(self):
        return [4, 3, 518, 518]
    
    def get_output_shape(self):
        return [4, 1000]
    
    def get_dtype(self):
        return "float"
    
    def get_trainloader(self):
        
        transform = transforms.Compose([
            transforms.RandomResizedCrop(518),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225])
        ])
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform)
        
        train, val = torch.utils.data.random_split(trainset, [0.1, 0.9])

        train_loader = DataLoader(train, batch_size=4, shuffle=True)
        return train_loader
    
    def get_mha_num_heads(self):
        return 16
    
    def get_mha_embed_dim(self):
        return 1280
    
    def get_dropout(self):
        return 0.0
    
    def get_batch_frist(self):
        return True


def training_normal(model, trainloader, device, optimizer, loss_fn):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        input, label = data
        input, label = input.to(device), label.to(device)

        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, label.squeeze())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    

def training_pipeline(model, trainloader, nb_gpu, optimizer, loss_fn):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        input, label = data
        
        input = input.to(0)
        label = label.to(nb_gpu - 1)

        optimizer.zero_grad()

        output = model(input).local_value()
        loss = loss_fn(output, label.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
