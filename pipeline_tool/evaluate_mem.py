#    Copyright (C) 2023  Bruno Da Rocha Carvalho, Gabriel Catel Torres Arzur
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from gpu_alloc import TraceMalloc
from dataset import PipelineDataset
from pipelinecache.layered_model import PipelinedModel
import os
import time
import torch
import argparse
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29600'

parser = argparse.ArgumentParser()
parser.add_argument('--input_shape', type=str, help='Input shape as a list')
parser.add_argument('--output_shape', type=str, help='Output shape as a list')
parser.add_argument('--number_gpu', type=int, help='Number of GPU')
parser.add_argument('--number_chunks', type=int, help='Number of chunks')
parser.add_argument('--dtype', type=str,  help='Type of output\'s tensor (long, float32...)')
args = parser.parse_args()

input_shape = args.input_shape.replace("[", "").replace("]", "")
input_shape = input_shape.split(",")
input_shape = [int(x.strip()) for x in input_shape]

output_shape  = args.output_shape.replace("[", "").replace("]", "")
output_shape  = output_shape.split(",")
output_shape  = [int(x.strip()) for x in output_shape]

number_gpus   = args.number_gpu
number_chunks = args.number_chunks


trace_gpu_alloc = TraceMalloc(number_gpus)
criterion = torch.nn.CrossEntropyLoss()

torch.cuda.init()
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

with trace_gpu_alloc:

    model = PipelinedModel()
    dataset = PipelineDataset(1024, input_shape[1:], [1] if len(output_shape) == 1 else output_shape[1:], args.dtype)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=input_shape[0], shuffle=True)

    model = model.get_modules()
    model = torch.distributed.pipeline.sync.Pipe(model, number_chunks)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in range(3):
        start_time = time.time()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(0)
            labels = labels.to(number_gpus- 1)

            try:
                outputs = model(inputs).local_value()
            except Exception as e:
                print(e)
                exit()

            labels_tmp = labels.squeeze()
            
            loss = criterion(outputs, labels.squeeze())
                
            loss.backward()
        end_time = time.time()
        execution_time = end_time - start_time

print(trace_gpu_alloc.peaked)


