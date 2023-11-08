import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Define the project root and add it to the sys path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)

# Define the CNN model
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

# Create the CNN model and save a reference to it
model = CNN()
model_saved = model

# Prepare configuration
from pipeline_tool.pipeline_config import PipelineConfig
# from pipeline_config import PipelineConfig
batch_size = 4

# Define the input and output shapes and data type
config_pipeline = PipelineConfig(input_shape=[batch_size, 3, 32, 32],
                                 output_shape=[batch_size],
                                 data_type="long")

# Prepare pipelined model with skippable tracing
from pipeline_tool.pipeline_tool import SkippableTracing

nb_gpu = torch.cuda.device_count()
trace = SkippableTracing(nb_gpus=nb_gpu, model=model, config=config_pipeline)

# Get modules from tracing
model = trace.get_modules()

# Prepare Pipe from API torch
from torch.distributed.pipeline.sync import Pipe
nb_chunk = 2

# Initialize RPC
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29600'
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

# Create a pipelined model
model = Pipe(module=model, chunks=nb_chunk)

# Download the CIFAR-10 dataset
import torchvision
import torchvision.transforms as transforms

# Define data transformations
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Load the test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Train the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        input, label = data

        input = input.to(0)
        label = label.to(nb_gpu - 1)

        optimizer.zero_grad()

        output = model(input).local_value()

        loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Validate the model
correct = 0
total = 0

# Since we're not training, we don't need to calculate gradients for outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(0)
        labels = labels.to(nb_gpu - 1)
        # Calculate outputs by running images through the network
        outputs = model(images).local_value()
        # The class with the highest energy is the prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# Save the model
trained_weights = {}
for (_, value_src), (key, _) in zip(model.state_dict().items(), model_saved.state_dict().items()):
    trained_weights[key] = value_src

model_saved.load_state_dict(trained_weights)

# Define the path to save the model
PATH = './cifar_net.pth'
torch.save(model_saved.state_dict(), PATH)
