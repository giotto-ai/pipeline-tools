import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from pipeline_tool.pipeline_config import PipelineConfig
from pipeline_tool.pipeline_tool import SkippableTracing

# Define the Vision Transformer model
from torchvision.models import vit_h_14
class VisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = vit_h_14(weights='DEFAULT')
    
    def forward(self, image):
        return self.model(image)
    

model = VisionTransformer()
batch_size = 4
config_pipeline = PipelineConfig(input_shape=[batch_size, 3, 518, 518],
                                 output_shape=[batch_size, 1000],
                                 data_type="float")

nb_mha = 33
num_heads = 16
embed_dim = 1280
dropout = 0.0
batch_first = True

config_pipeline.create_mha_conf_equal(nb_mha, num_heads, embed_dim, dropout, batch_first)

def test_trace():
    trace = SkippableTracing(nb_gpus=0, model=model, config=config_pipeline)
