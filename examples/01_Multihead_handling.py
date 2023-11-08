import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Define the project root and add it to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Define the Vision Transformer model
from torchvision.models import vit_h_14
class VisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = vit_h_14(weights='DEFAULT')
    
    def forward(self, image):
        return self.model(image)
    
# Create the Vision Tranformer model and save a reference to it
model = VisionTransformer()
model_saved = model

# Prepare configuration
from pipeline_tool.pipeline_config import PipelineConfig
batch_size = 4

# Define the input and output shapes and data type
config_pipeline = PipelineConfig(input_shape=[batch_size, 3, 518, 518],
                                 output_shape=[batch_size, 1000],
                                 data_type="float")

# Add Multihead configuration to PipelineConfig
nb_mha = 33
num_heads = 16
embed_dim = 1280
dropout = 0.0
batch_first = True

config_pipeline.create_mha_conf_equal(nb_mha, num_heads, embed_dim, dropout, batch_first)

# Prepare pipelined model with skippable tracing
from pipeline_tool.pipeline_tool import SkippableTracing

# Here it should end if you have not enough space on your GPU to handle it by an error CUDA OOM.
nb_gpu = 1
try:
    trace = SkippableTracing(nb_gpus=nb_gpu, model=model, config=config_pipeline)
except Exception as e:
    print(e)
    # Change to two or more GPU to be able to use this model
    nb_gpu = 2
    try:
        trace = SkippableTracing(nb_gpus=nb_gpu, model=model, config=config_pipeline)
    except Exception as e:
        print(e)

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
from torchvision import transforms
transform = transforms.Compose([
            transforms.RandomResizedCrop(518),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225])
        ])

# Load the training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

train, val = torch.utils.data.random_split(trainset, [0.1, 0.9])

trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                          shuffle=True)


# Load the test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test, val = torch.utils.data.random_split(testset, [0.1, 0.9])

testloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=False)

# Train the model with not enough space
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