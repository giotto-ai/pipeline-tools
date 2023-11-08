import argparse
import torch
import torch.nn as nn
import utils
import os
import sys
import time
from pipeline_tool.pipeline_config import PipelineConfig
from pipeline_tool.pipeline_tool import SkippableTracing

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29600'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from torch.distributed.pipeline.sync import Pipe

nb_epochs = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mode:
    def __init__(self, model, chunk, framework, gpu, epochs):
        self.model     = model
        self.chunk     = chunk
        self.framework = framework
        self.gpu       = gpu
        self.epochs    = epochs

    def set_model(self, model):
        self.model = model

    def set_chunk(self, chunk):
        self.chunk = chunk

    def set_gpu(self, gpu):
        self.gpu = gpu

    def set_framework(self, framework):
        self.framework = framework


class BenchmarkMode:
    def __init__(self, args):
        self.args = args
        self.mem_alloc = []
        self.exec_time = []
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss_fn()

    def setup_model(self):
        count_mha = 1
        if self.args.model == "CNN":
            self.model = utils.CNN()
        elif self.args.model == "FFNET":
            self.model = utils.FFNET()
        elif self.args.model == "Transformers":
            self.model = utils.Transformers()
        elif self.args.model == "Basic":
            self.model = None
        elif self.args.model == "Debug":
            self.model = utils.Debug()
        elif self.args.model == "BigModel":
            self.model = utils.BigModel()
            for name, module in self.model.named_modules():
                if str(module).split('(', 1)[0].find('Multi') >= 0:
                    count_mha += 1
        else:
            raise ValueError("Given model is not known.")

        input_shape = self.model.get_input_shape()
        output_shape = self.model.get_output_shape()

        self.trainloader = self.model.get_trainloader()
        if count_mha > 1:
            config = PipelineConfig.video_transform(PipelineConfig)
        else:
            config = PipelineConfig(input_shape, output_shape, self.model.get_dtype())

        if self.args.framework == "Pipeline": 
            trace = SkippableTracing(self.args.gpu, self.model, config)
            torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
            self.model = trace.get_modules()
            self.model = Pipe(self.model, chunks=self.args.chunk)

        elif self.args.framework == "API torch":
            self.model.to(device)

    def setup_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def setup_loss_fn(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def run(self):
        for epoch in range(self.args.epochs):
            start  = [0] * self.args.gpu
            peaked = [0] * self.args.gpu

            for gpu in range(self.args.gpu):
                start[gpu] = torch.cuda.memory_allocated(gpu)
            
            start_time = time.time()

            if self.args.framework == "Pipeline":
                utils.training_pipeline(self.model, self.trainloader, self.args.gpu, self.optimizer, self.loss_fn)

            elif self.args.framework == "API torch":    
                utils.training_normal(self.model, self.trainloader, device, self.optimizer, self.loss_fn)
            
            end_time = time.time()

            execution_time = end_time - start_time

            for gpu in range(self.args.gpu):
                peaked[gpu] = (torch.cuda.max_memory_allocated(gpu) - start[gpu]) // (2 * 1024)

            self.mem_alloc.append(peaked)
            self.exec_time.append(execution_time)
    
    def generate_stats_string(self):
        string = f"{self.args.framework};{self.args.model};{self.args.gpu};{self.args.chunk};"

        for time in self.exec_time:
            string += f"{time};"

        for alloc in self.mem_alloc:
            string += f"{alloc};"

        string = string[:-1]
        string += "\n"
        return string

def main():
    parser = argparse.ArgumentParser(description="Script d'analyse avec différentes options")

    parser.add_argument(
        "model",
        choices=["CNN", "FFNET", "Basic", "Transformers", "Debug", "BigModel"],
        help="Modèle à utiliser (CNN, FFNET, Basic, Transformers)"
    )
    parser.add_argument(
        "framework",
        choices=["Pipeline", "API torch"],
        help="Choix du framework (Pipeline, API torch)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="Nombre de GPU à utiliser (par défaut 1)"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=2,
        help="Nombre de chunks (par défaut 2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Nombre d'époques (par défaut 10)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Result directory"
    )

    args = parser.parse_args()

    mode = Mode(args.model, args.chunk, args.framework, args.gpu, args.epochs)

    bench = BenchmarkMode(mode)
    
    bench.run()

    with open(f"{args.dir}/results.txt", "a") as f:
        f.write(bench.generate_stats_string())

if __name__ == "__main__":
    main()