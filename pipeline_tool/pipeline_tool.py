#    The Pipeline tool allows you to create a new model, split across multiple GPUs, 
#      from a PyTorch module. This enables the training of large models that do not fit on a single GPU.
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

from torch.fx import symbolic_trace
import torch
from pathlib import Path
from .constant import TAB
import subprocess
import sys
from .class_impl.CallModule import CallModule
from .class_impl.CallFunction import CallFunction
from .class_impl.PropagationLayer import PropagationLayer
from .class_impl.GetAttr import GetAttr
from .class_impl.GetAttrModule import GetAttrModule
import logging

class SkippableTracing:
    """Create and sequence the model parallelism.

    He will parsed the model given by the user and generate a file that contain all the splitted Layer and divide them on the gpus.

    Example of use :
    .. code-block:: python
        trace = SkippableTracing(2, model)
        # Then just get the generated splitted model
        model = trace.get_modules()

    :param nb_gpus: Nb of gpus of work, if none given max gpus will be taken.
    :type nb_gpus: int
    :param model: Model to parsed
    :type model: a model extended of nn.Module.
    """
    def __init__(self, nb_gpus, model, config):
        """Constructor."""
        self.module_desc = {}
        self.file = ""
        self.net = None
        self.LayerLists = {}
        self.GetattrLists = {}
        self.nb_gpu = nb_gpus if nb_gpus is not None else torch.cuda.device_count()
        self.file_name = "layered_model.py"
        self.directory_name = "pipelinecache"
        self.configs_mha = config.config_mha
        self.mha_number = len(self.configs_mha) 
        self.mha_count = 0
        self.input_shape = config.input_shape
        self.output_shape = config.output_shape
        self.dtype = config.data_type

        # logging.basicConfig(filename='pipelinecache/debug.log', encoding='utf-8', level=logging.DEBUG)
        
        self._verfiy_config_mha()

        self._tracer(model)

    def _verfiy_config_mha(self):
        """Verify if at least embed_dim and num_heads are present in the configuration given by the user.
        
        This two parameters are mandatory to create a MHA.
        """
        for config in self.configs_mha:
            try:
                config['embed_dim']
            except KeyError as e:
                raise KeyError("You didn't provide embed_dim in one of your MHA config")
            
            try:
                config['num_heads']
            except KeyError as e:
                raise KeyError("You didn't provide num_heads in one of your MHA config")

    def _write_in_file(self):
        """Write to the output file the generated Layers to use it from other files."""        
        dir_path = Path(__file__).resolve().parent / self.directory_name

        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        file_path = dir_path / self.file_name

        with open(file_path, "w") as f:
            f.write(self.file)
            f.close()

    def get_modules(self) -> torch.nn.Sequential:
        """Allow the user to get the generated Sequential model for each GPU."""
        from .pipelinecache.layered_model import PipelinedModel

        model = PipelinedModel()
        return model.get_modules()

    def _init_file(self):
        """Add all necessary import to the file."""
        self.file += "import torch\n"
        self.file += "import torch.nn.functional as F\n"
        self.file += "import torch.nn as nn\n"
        self.file += "from torch.distributed.pipeline.sync.skip import stash, pop, skippable \n\n"

    def _generate_end_class(self):
        """Add a class at the end of the generated file to get simply the pipelined model."""
        self.file += f"class PipelinedModel(nn.Module):\n"
        self.file += TAB[1] + "def __init__(self) -> None:\n"
        self.file += TAB[2] + "super().__init__()\n"

        gpu_index = 0
        self.file += TAB[2] + f"self.s{gpu_index} = nn.Sequential("

        for layer in self.LayerLists.values():
            self.file += f"{layer.get_name()}_layer(), "

            if layer.get_separation_layer():
                self.file = self.file[:-2]
                self.file += f").cuda({gpu_index})\n"
                gpu_index = gpu_index + 1
                self.file += TAB[2] + f"self.s{gpu_index} = nn.Sequential("

        self.file = self.file[:-2]
        self.file += f").cuda({gpu_index})\n"

        self.file += TAB[1] + "def forward(self, input):\n"
        self.file += TAB[2] + f"ret = input\n"
        for gpu in range(self.nb_gpu):
            self.file += TAB[2] + f"ret = self.s{gpu}(ret.to({gpu}))\n"

        self.file += TAB[2] + "return ret\n"

        self.file += TAB[1] + "def get_modules(self):\n"
        self.file += TAB[2] + "return  nn.Sequential(*["
        for gpu in range(self.nb_gpu):
            self.file += f"nn.Sequential(*self.s{gpu}),"
        self.file = self.file[:-1]
        self.file += "])\n"

    def _create_mha(self):
        """Create MHA string declaration."""
        config = self.configs_mha[self.mha_count]

        decl = f"MultiheadAttention("

        for key, param in config.items():
            decl += f"{key}={param}, "

        decl = decl[:-2]
        decl += ")"
        self.mha_count = self.mha_count + 1

        return decl

    def _catch_module_desc(self):
        """Create a look-up dictionary to match target names with their declaration.
        
        We use the withelist.txt to know which module name we have to keep as "core" module.
        All the modules not present in the whitelist willbe digged to found their composition.

        MultiheadAttention are not parsed, so we have to do a little trick to found how is it configured, only based on giotto_deep implementation.
        """
        filename = Path(__file__).resolve().parent / 'whitelist.txt'
        whitelist = open(filename).readlines()
        whitelist = [line.strip() for line in whitelist]

        for name, module in self.net.named_modules():
            if str(module).split('(', 1)[0] in whitelist:
                if str(module).split('(', 1)[0].find('Multi') >= 0:
                    # try:
                    if self.mha_number >= 1:
                        self.module_desc[name] = self._create_mha()
                    else:
                        raise UserWarning(f"Error: You didn't specified any MHA config, but at least one exist.")
                    # except UserWarning as e:
                    #     raise Exception(f"Error : {e}")
                else:
                    self.module_desc[name] = module

    def _balancing(self, layers, memory):
        """Balance the distribution of layers across GPUs based on memory usage.
    
        :param layers: Current distribution of layers across GPUs.
        :type layers: list
        :param memory: Memory usage for each GPU.
        :type memory: list
        :return: New balanced distribution of layers across GPUs.
        :rtype: list
        """
        repartition = layers.copy()
        memory_tmp = memory.copy()

        # We couldn't have only 1 layer on first or last GPU, so we remove it to avoid 
        #   this case.
        repartition[0]  -= 1
        repartition[-1] -= 1
        
        n = len(layers)

        lower_idx = min(range(n), key=lambda i: memory_tmp[i])

        while True:
            upper_idx = max(range(n), key=lambda i: memory_tmp[i])
            memory_tmp[upper_idx] = 0
            if repartition[upper_idx] > 1:
                break
        
        repartition[lower_idx] += 1
        repartition[upper_idx] -= 1
        
        # We restablish the two deleted layers.
        repartition[0]  += 1
        repartition[-1] += 1


        return repartition

    def reset_repartition(self, layer_per_gpu):
        """Reset the distribution of layers based on the number of layers per GPU.
    
        :param layer_per_gpu: Number of layers per GPU.
        :type layer_per_gpu: list
        """
        current_layer = 0
        gpu_index = 0
        separation_layer_index = layer_per_gpu[gpu_index] - 1

        for _, layer in self.LayerLists.items():
            if current_layer >= len(self.LayerLists.items()) - 1:
                break
            
            if separation_layer_index == current_layer:
                layer.reset_separation_layer()
                gpu_index += 1
                separation_layer_index += layer_per_gpu[gpu_index]

            current_layer += 1

    def set_repartition(self, layer_per_gpu):
        """Set the distribution of layers across GPUs based on the number of layers per GPU.
    
        :param layer_per_gpu: Number of layers per GPU.
        :type layer_per_gpu: list
        """
        current_layer = 0
        gpu_index = 0
        separation_layer_index = layer_per_gpu[gpu_index] - 1

        for _, layer in self.LayerLists.items():

            if current_layer >= len(self.LayerLists.items()) - 1:
                self.file += layer.get_declaration()
                break
            
            if separation_layer_index == current_layer:
                layer.set_separation_layer()
                gpu_index += 1
                separation_layer_index += layer_per_gpu[gpu_index]

            self.file += layer.get_declaration()
            current_layer += 1

        self._generate_end_class()

    def _check_memory_peak(self, memory_peak):
        """Check if the memory peaks are balanced across GPUs.
    
        :param memory_peak: Memory peaks for each GPU.
        :type memory_peak: list
        :return: True if memory peaks are balanced, False otherwise.
        :rtype: bool
        """
        threshold = 0.2 
        reference_value = memory_peak[0]
        return all(abs(value - reference_value) <= reference_value * threshold for value in memory_peak[1:])


    def _repartition(self):
        """Perform the distribution of layers across GPUs in a balanced manner based on memory usage."""
        self._init_file()
        # Save self var for remake
        file = self.file
        # Calculate first naive repartition on gpus
        clone_step = len(self.LayerLists.items()) // self.nb_gpu
        remainder = len(self.LayerLists.items()) % self.nb_gpu
        layer_per_gpu = [clone_step] * self.nb_gpu

        # Distribute remainder layer to the GPU
        for i in range(remainder):
            layer_per_gpu[i] += 1

        # Initialise cloned layers
        self.set_repartition(layer_per_gpu)

        # Write in file the naive repartition
        self._write_in_file()

        dir_path = Path(__file__).resolve().parent / "evaluate_mem.py"

        previous_repartitions = []

        while True:
            if layer_per_gpu[0] == 1 or layer_per_gpu[:-1] == 1:
                raise Exception(f"The model is not separable in {self.nb_gpu} GPU, first or last GPU contain only 1 layer. {layer_per_gpu}")

            if 0 in layer_per_gpu:
                raise Exception(f"At least one GPU have 0 layer, please allocate less GPU. {layer_per_gpu}")

            p = subprocess.run([sys.executable, dir_path,
                            '--input_shape', str(list(self.input_shape)),
                            '--output_shape', str(list(self.output_shape)),
                            '--number_gpu', str(int(self.nb_gpu)),
                            '--number_chunks', str(2),
                            '--dtype', str(self.dtype)], capture_output=True, text=True)

            result = p.stdout
            if "CUDA" in result:
                raise Exception(f"The model is too big for the number of GPU given. CUDA OOM")
            
            elif result == '':
                raise Exception(f"An error occured during training. The repartition tried was {layer_per_gpu}.")

            result = result.replace("[", "").replace("]", "")
            result = result.split(",")
            memory_peak = [int(x.strip()) for x in result]
            
            if not self._check_memory_peak(memory_peak):
                new_layer_per_gpu = self._balancing(layer_per_gpu, memory_peak)
                if new_layer_per_gpu in previous_repartitions:
                    break
                
                previous_repartitions.append(new_layer_per_gpu)

                self.file = file
                self.reset_repartition(layer_per_gpu)
                self.set_repartition(new_layer_per_gpu)
                self._write_in_file()
                layer_per_gpu = new_layer_per_gpu

            else:
                break
    
    

    def _filter_trace(self, trace):
        """Filter trace by removing blacklisted layers and un-propaged layers.

        :param trace: Trace of the model
        :type trace: torch.fx.graph._node_list
        """
        to_remove = []        
        filename = Path(__file__).resolve().parent / 'blacklist.txt'
        blacklist = open(filename).readlines()
        blacklist = [line.strip() for line in blacklist]

        for node in trace.graph.nodes:
            if any(item in node.name for item in blacklist):
                to_remove.append(node)

        for node in to_remove[::-1]:
            trace.graph.erase_node(node)

        all_unused_deleted = True
        while all_unused_deleted == True:
            all_unused_deleted = False
            for node in trace.graph.nodes:
                if node.name == "output":
                    break 
                try:
                    trace.graph.erase_node(node)
                    all_unused_deleted = True
                except:
                    continue


    def _tracer(self, net):
        """Trace and create all the composite needed to describe correctly the models given.
        
        It will call the class of class_impl folder to generate at the end the correct file of the model splited between GPUs.
        :param net: Model to trace.
        :type net: a model extended of nn.Module.
        """
        self.net = net

        try:
            trace = symbolic_trace(net)
        except Exception as e:
            raise Exception(f"The model given cannot be traced by torch.fx. Error : {e}")
        
        self._filter_trace(trace)

        self._catch_module_desc()

        prev_node = None

        # Iter through each node traced by torch.fx
        for node in trace.graph.nodes:
            if str(node).find("getitem") >= 0:
                for _node in trace.graph.nodes:
                    if node in _node.args:
                        if str(node.args[0]).find("getattr") >= 0:
                            if self.GetattrLists[node.args[0]].getitem_idx is None:
                                self.GetattrLists[node.args[0]].add_getitem(node.args[1])
                            else:
                                self.GetattrLists[node] = GetAttr(node.args[0], trace.graph.nodes, _node)
                                self.GetattrLists[node].add_getitem(node.args[1])
                            
                        else:
                            self.LayerLists[node.args[0]].add_getitem(node.args[1])
                            
            elif str(node).find("getattr") >= 0:
                self.GetattrLists[node] = GetAttr(node, trace.graph.nodes)

            else:
                if node.op == "call_module":
                    self.LayerLists[node] = CallModule(node, trace.graph.nodes, prev_node,
                                                       self.module_desc[node.target])

                elif node.op == "call_function" or node.op == "call_method":
                    self.LayerLists[node] = CallFunction(node, trace.graph.nodes, prev_node)
                elif node.op == "get_attr":
                    self.LayerLists[node] = GetAttrModule(node, trace.graph.nodes, prev_node, net)
                    pass

                else:
                    self.LayerLists[node] = PropagationLayer(node, trace.graph.nodes, prev_node)
                prev_node = node

        # For each getattr, we will update the Layer who are linked to it. If the value of the getattr need to be
        # stashed we will update the stash list of the parent of the getattr. And by default update all the argument
        # to have the correct declaration with the getattr.
        for _, getattr_item in self.GetattrLists.items():
            # if getattr_item.get_child() is not None:
            if getattr_item.is_stash_needed():
                self.LayerLists[getattr_item.get_parent()].add_stash(getattr_item.get_child())

            self.LayerLists[getattr_item.get_child()].update_arg_by_attr(getattr_item.get_attr_name(), getattr_item.get_position())

        # As it is complicated to trace efficiently the stash we updated it with the poplist of each node.
        # So for each "pop parent" we set a stash for the current node.
        for layer in self.LayerLists.values():
            for pop_parent in layer.get_pop_parent():
                self.LayerLists[pop_parent[0]].set_stash(layer.get_node())


        # Test mode without GPUs
        if self.nb_gpu == 0:
            print("Pipeline tool as performed a full tracing, but is not allowed to use GPU.")
        else:
            self._repartition()