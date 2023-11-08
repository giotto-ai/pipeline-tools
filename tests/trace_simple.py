import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from pipeline_tool.pipeline_config import PipelineConfig
from pipeline_tool.pipeline_tool import SkippableTracing

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

model = CNN()



batch_size = 4

config_pipeline = PipelineConfig(input_shape=[batch_size, 3, 32, 32],
                                 output_shape=[batch_size],
                                 data_type="long")

trace = SkippableTracing(nb_gpus=0, model=model, config=config_pipeline)