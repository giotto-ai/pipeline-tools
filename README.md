# Pipeline tool

## Table of contents

[toc]

## Introduction

The field of machine learning is constantly evolving, with increasingly sophisticated models and ever-expanding datasets. Significant challenges can arise for professionals in the field, especially when it comes to training models that are too large to fit into the memory of a single GPU.

In this context, a tool has been developed to distribute a PyTorch machine learning model across multiple GPUs without altering the training process. The PyTorch model description is taken as input, each layer is interpreted independently, and the model is rewritten to handle the operations interdependencies. The result is a new model that can be automatically distributed (creating a pileline) across multiple GPUs that does not affects the results quality.

In the following, this tool will be presented in detail, along with the benefits it can provide to machine learning professionals seeking to train large and complex models.

## How it works

### First step

To run the tool, you have to provide some parameters:

- Number of GPUs: If the user does not specify the number of GPUs to use, the tool will automatically detect the available GPUs on the machine running the command. In this case, the model will be trained using all detected GPUs to improve performance.
- PyTorch Model: The user must provide a PyTorch model that only uses functions and modules coming from the PyTorch API. In other words, the model should not incorporate custom functions unknown to the PyTorch API. However, it is entirely possible to create custom layers using combinations of functions (always from the PyTorch API).
- Shapes of the input and output: This will be needed to profile [memory usage](#model-splitting).
- Data type that will be passed in the model, for example float.

The first step is adding the following imports in your project:

```python
from torch.distributed.pipeline.sync import Pipe 
from pipeline_tool.pipeline_tool import SkippableTracing
from pipeline_tool.pipeline_config import PipelineConfig
```

Next, you need to use the PipelineConfig class, which allows you to prepare the necessary parameters (input and output shape, data type).

```python
 config = PipelineConfig([X, Y, Z], [X, Y], "dtype")
```
> *One important thing to know is that the first number given in input/output shape is your batch size.*

Once you have defined your config and create your model, you can process it as show in the example bellow.

```python
N_GPUs = 2
trace = SkippableTracing(N_GPUs, model, config)
graph_model = trace.get_modules()
```

Here the model is traced using [torch.fx](https://pytorch.org/docs/stable/fx.html) to obtain the GraphModule. This allows us to determine, for each module, its type, parameters, functions (e.g., convolution, activation, multiplication) and their links to others modules.

Bellow an example of how is treated a simple model : 

![03_simple_model_dep](img/03_simple_model_dep.png)

In this basic example, we have a model composed exclusively of PyTorch modules. To describe them accurately, we utilize the trace generated by torch fx. 

The generated trace appears as follows:

```bash
Opcode          Name            Target         
placeholder     x               x              
call_module     linear1         linear1        
call_module     activation      activation     
call_module     linear2         linear2        
call_module     softmax         softmax        
output          output          output  
```

This trace allows us to identify each generated layer and provides the following information:

- Opcode: Indicates the type of operation performed by the layer.
- Name: Corresponds to the name of the function or operation performed by the layer.
- Target: Represents the name of the layer as it appears in the description of the PyTorch model.

Thus, the trace provides a detailed view of the operations performed by each layer, making it easier to understand and analyze the model.

```bash
Name            Module declaration
linear1         Linear(in_features=100, out_features=200, bias=True)
activation      ReLU()         
linear2         Linear(in_features=200, out_features=10, bias=True)
softmax         Softmax(dim=None)
```

The retrieval, analysis, and management of all this information enable the generation of a file containing a new model ready for pipelined training on N GPUs.

### Complex Models

Unfortunately, a model is never limited to a simple linear sequence of modules taking the output of the previous operation as input... More complex models exist, and it is necessary to handle all possible cases, to trace the model correctly so that it is faithfully reproduced without omitting certain operations.

As a result, it is necessary to distinguish PyTorch modules from other elements.

We analyze the model received as a parameter and store the elements by their names in a dictionary, which we use to create a correspondence table with the names given by the trace.

We can then iterate over the generated trace to differentiate five types of layers:

1. <u>**Module**</u>: These need to be initialized and thus require their description to be retrieved from the original model.
2. <u>**Function**</u>: These correspond to simple PyTorch functions executed between tensors or on a tensor (e.g., additions, dimension reductions, etc.).
3. <u>**Getitem**</u>: These appear in the trace when only a portion of a result in the form of a list needs to be retrieved (e.g., index 1 of a list or the first return value of a function).
4. <u>**Getattr**</u>: These correspond to retrieving an attribute of a tensor.
5. <u>**Propagation**</u>: These appear in the trace to propagate tensors to other layers.

#### call_function

Let's explore the concept of call_function with the widely known ResNet model.

When we examine the generated trace, we notice a new opcode, distinct from the one we previously discussed in the [first step](#first-step) : 

```bash
Opcode               Name                           Arguments                     
placeholder          x                              ()                            
call_module          flow_0_conv1                   (x,)                          
[...]  
call_module          flow_0_avgpool                 (flow_0_layer4_1_relu_1,)   
# ############################################################################################
call_function        flatten                        (flow_0_avgpool, 1) 
# ############################################################################################
call_module          flow_0_fc                      (flatten,)                    
call_module          flow_1                         (flow_0_fc,)                  
output               output                         (flow_1,)    
```
*Notice that we also have access to the input arguments of each layer.*

call_functions are treated differently from call_modules and consequently generate distinct code. Therefore, each call_function is declared as a Torch module that exclusively performs the necessary operation. In the case of the previous trace, let's consider the declaration of the call_function `flatten`:

```python
class flatten_layer(nn.Module):
    def forward(self, input):
        ret = torch.flatten(input, 1)
        return ret
```

Functions do not necessitate an initialization function. Instead, our tool seeks out the appropriate Torch function based on the name provided in the trace. For instance, when working with the instantiated ResNet18 model, the function "flatten" already exists within the Torch API.

The trace allows us to identify the arguments passed to this function. In the case above, the inputs are the output of the previous layer and the integer "1".

#### Propagation

As discussed in the section on [complex models](#complex-models) there are instances where we need to transmit the output of one layer to others that are not inherently connected to it. To facilitate this process, PyTorch provides a useful decorator called "skippable." This decorator introduces two key features:

1. `stash`: This feature permits us to store a specific value with an associated name, allowing for convenient retrieval later.

2. `pop`: With this functionality,

Let's get a look into an example trace to have a better understanding::

```bash
Opcode               Name                           Arguments                     
placeholder          x                              ()                            
call_module          flow_0_conv1                   (x,)                          
[...]              
call_module          flow_0_maxpool                 (flow_0_relu,)                
call_module          flow_0_layer1_0_conv1          (flow_0_maxpool,)             
call_module          flow_0_layer1_0_bn1            (flow_0_layer1_0_conv1,)      
call_module          flow_0_layer1_0_relu           (flow_0_layer1_0_bn1,)        
call_module          flow_0_layer1_0_conv2          (flow_0_layer1_0_relu,)       
call_module          flow_0_layer1_0_bn2            (flow_0_layer1_0_conv2,) 
#############################################################################################
call_function        add                            (flow_0_layer1_0_bn2, flow_0_maxpool)
#############################################################################################
call_module          flow_0_layer1_0_relu_1         (add,)                        
[...]          
call_module          flow_0_fc                      (flatten,)                    
call_module          flow_1                         (flow_0_fc,)                  
output               output                         (flow_1,)    
```

The call_function surrounded have two name in input : 
- flow_0_layer1_0_bn2, which directly stems from the previous layer.
- flow_0_maxpool, originating from an earlier layer in the model.

Our tool is designed to establish connections between layers and retain information about the arguments derived from prior layers. 

Consequently, when utilizing the skippable decorator in the generated code:

```python
[...]

@skippable(stash=['flow_0_maxpool_to_add'])
class flow_0_maxpool_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    def forward(self, input):
        ret = self.fc(input)
        yield stash('flow_0_maxpool_to_add', ret)
        return ret

[...]

class flow_0_layer1_0_bn2_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

@skippable(pop=['flow_0_maxpool_to_add'])
class add_layer(nn.Module):
    def forward(self, input):
        flow_0_maxpool = yield pop('flow_0_maxpool_to_add')
        ret = torch.add(input, flow_0_maxpool)
        return ret
    
[...]
```
We ensure that the dependencies between layers are properly preserved.


#### Getitem

Within the trace, certain call_function entries contain the term "getitem" in their names. This indicates that these are not conventional functions but rather indicate the need to access a specific index within a result. Consider the following trace as an example:

```bash
[...]
call_function        getattr_1            (add_3, 'shape')              
call_function        getitem_4            (getattr_1, 0)                
[...]
```

Here, we notice the presence of a getitem operation, which is applied to the result of the previous layer. If we were to translate this trace, it would resemble something like add_3.shape[0] (for an explanation of getattr, please refer to the [next point](#getattr)).

The challenge with getitem lies in the limitation of the Torch API, which does not allow the propagation of non-tensor values. Consequently, we must concatenate the getitem operation to the layer from which we require the value, rather than creating an independent layer that cannot effectively transmit its output.

#### GetAttr

There are two distinct types of `getattr` operations: 

1. **call_function with the Name "getattr"**: These instances occur when an attribute of modules needs to be accessed..
   
   In the provided trace:

   ```bash
   [...]
   call_function        getattr_1                                         (add_3, 'shape') 
   [...]
   call_module          model_pooling_layer_scaled_dot_product_attention  (expand, add_3, add_3)  
   ```

   As previously mentioned, we cannot propagate non-tensor values. The presence of getattr indicates the need to access a specific attribute within a module. In the trace above, the tensor add_3 possesses an attribute "shape" that will be utilized. In such cases, we refrain from creating new modules; instead, we reference the relevant attribute of the tensor when it is passed as a parameter.

   Here's an illustrative example of generated code to elucidate this approach:

   ```python
   [...]
   @skippable(stash=['add_3_to_expand', 'add_3_to_model_pooling_layer_scaled_dot_product_attention'], pop=['add_2_to_add_3'])
   class add_3_layer(nn.Module):
       def forward(self, input):
           add_2 = yield pop('add_2_to_add_3')
           ret = torch.add(input, add_2)
           yield stash('add_3_to_expand', ret)
           yield stash('add_3_to_model_pooling_layer_scaled_dot_product_attention', ret)
           return ret
   [...]
   @skippable(pop=['add_3_to_expand'])
   class expand_layer(nn.Module):
       def forward(self, input):
           add_3 = yield pop('add_3_to_expand')
           ret = input.expand(add_3.shape[0], -1, -1)
           return ret
   ```


2. **get_attr with the Opcode "get_attr"**: These occurrences arise when a private attribute of a user-created class is requested.

   In the provided trace:
   ```bash
   get_attr       model_pooling_layer_query           ()
   ```

   We only have the name of the attribute, and it needs to be initialized to propagate or utilize it, we create a module that initializes the attribute based on the provided information. We search for the attribute on the given model and recreate it identically.

   Here's an example of code to illustrate this process:
   
   ```python
   class model_pooling_layer_query_layer(nn.Module):
       def __init__(self) -> None:
           super().__init__()
           self.fc = nn.parameter.Parameter(torch.Tensor(1, 16), requires_grad=True)
       def forward(self, input):
           ret = self.fc
           return ret
   ```

#### MultiHeadAttention processing

Unpredictable management is, however, necessary for MultiHeadAttention. During the module declaration retrieval phase, it is impossible to retrieve those of the MultiHeadAttention. Therefore, the user must provide a dictionary containing the description of all the parameters and their values for the MultiHeadAttention of their model during the tool's initialization.

At a minimum, the following parameters must be provided for a MultiHead:

- embed_dim
- num_heads

And the initialization would be changed to three alternative:

1. Give your hand made dictionnary describing all your MHA

    ```python
    mha_config = [{'embed_dim': hidden_val, 'num_heads': heads_val, 'dropout': 0.1, 'batch_first': True},
                {'embed_dim': hidden_val, 'num_heads': heads_val, 'dropout': 0.1, 'batch_first': True},
                {'embed_dim': hidden_val, 'num_heads': heads_val, 'dropout': 0.1, 'batch_first': True},
                {'embed_dim': hidden_val, 'num_heads': heads_val, 'dropout': 0.1, 'batch_first': True},
                {'embed_dim': hidden_val, 'num_heads': heads_val, 'dropout': 0.1, 'batch_first': True}]
    config = PipelineConfig([X, Y, Z], [X, Y], "dtype", mha_config)
    nb_gpus = 2
    trace = SkippableTracing(nb_gpus, model, config)
    model_pipe = trace.get_modules()
    ```
2. Use MHA dictionnary generator in PipelineConfig class. If you know that all your MHA are identical, you can use this function to create N dictionnary entry identical.
    ```python
    config = PipelineConfig([X, Y, Z], [X, Y], "dtype")
    config.create_mha_conf_equal(embed_dim, num_heads, dropout, batch_first)
    nb_gpus = 2
    trace = SkippableTracing(nb_gpus, model, config)
    model_pipe = trace.get_modules()
    ```
3. Finaly some default model are setup with classmethod, Persformer, one bigger Persformer and VideoTransformer.
    ```python
    # TODO EXPLAINATION
    ```

### Model splitting

Now that we are capable of creating a model that can be distributed across multiple GPUs, the question arises: how do we split it intelligently? Currently, the tool proceeds with a somewhat naive approach. We create a dummy dataset to pass through the model and perform a training run. This allows us to measure the memory loads on all GPUs.

Initially, the tool divides the layers into two equal parts (in terms of the number of layers) and conducts these memory load measurements.
If the load is not evenly distributed, we re-write the model (moving layers around) and iterate the dummy run, until we achieving uniform  distribution on N GPUs.


## Pipeline Tool Example
Two examples are provided in `examples/`. It shows how : 

1. Use the pipeline_tool
2. Train a pipelined model
3. Evaluating a pipelined model
4. Save the trained model.


## Pipeline Tool x Giotto Deep
The Pipeline tool is seamlessly integrated into Giotto-Deep's trainer, requiring no changes to their API.

Here's an example of what need to be done: 

```python
# New import
from gdeep.trainer.trainer import Parallelism, ParallelismType

# Create the trainer as before
trainer = Trainer(wrapped_model, [dl_train, dl_train], loss_function, writer) 

# Prepare the config of the MHA
configs = [{'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
         {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
         {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
         {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
         {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True}]

# List of device
devices = list(range(torch.cuda.device_count()))

# Use the Parallelism class created to prepare the trainer for a pipeline run
parallel = Parallelism(ParallelismType.PIPELINE,
                       devices,
                       len(devices),
                       pipeline_chunks=2,
                       config_mha=configs)

# Call the train function with the new parameter
trainer.train(Adam, 2, parallel=parallel)
```

### Example
To experiment with Giotto Deep training using the Pipeline tool in your environment, two example scripts have been provided. Navigate to Giotto's examples folder and run either "pipeline_basic_image.py" or "pipeline_orbit5k.py" with the --pipeline argument to enable the pipeline mode, or without it for regular training.

## Installation

### Install from sources
Launch from the root of the project:

```bash
pip install .
```
It will install pipeline_tool on your python envrionement.

The import necessary are :
```python3
from pipeline_tool.pipeline_config import PipelineConfig
from pipeline_tool.pipeline_tool import SkippableTracing
```

## Benchmarking

A benchmarking tool has been made available. This script will test the pipeline_tool on 3 different models:

1. A FFNET
2. A CNN
3. One VisionTransformer

With these 3 models, we cover the majority of cases that the tool will have to deal with. The CNN and FFNET are two small models and will soon find it impossible to share too much, while the VisionTransformer is very large and doesn't necessarily fit on 1 GPU, but it also contains MultiHeads.
This is how we proceed: 
When the script is launched, we set the maximum number of GPUs in the environment (in the example below, 8), then run the first execution with the torch API to create a repository before launching the analyses with the pipeline_tool.

The results are as follows: 

|Framework|Model   |Number of GPUs|Number of Chunks|Time 1 [s]        |Time 2 [s]        |Time 3 [s]         |Time 4 [s]         |Alloc 1 [MB]                                                            |Alloc 2 [MB]                                                            |Alloc 3 [MB]                                                            |Alloc 4 [MB]                                                            |
|---------|--------|--------------|----------------|------------------|------------------|-------------------|-------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|
|API torch|CNN     |1             |0               | 1.99 |0.53|0.53 |0.54 |[747]                                                                   |[625]                                                                   |[625]                                                                   |[625]                                                                   |
|API torch|FFNET   |1             |0               |0.85 |0.25|0.24|0.25|[676]                                                                   |[520]                                                                   |[520]                                                                   |[520]                                                                   |
|Pipeline |CNN     |1             |2               |2.73|1.16|1.22 |1.28  |[844]                                                                   |[698]                                                                   |[698]                                                                   |[698]                                                                   |
|Pipeline |CNN     |2             |2               |3.4|2.03|2.05  |2.14 |[706, 537]                                                              |[582, 514]                                                              |[582, 514]                                                              |[582, 514]                                                              |
|Pipeline |CNN     |3             |2               |4.24|2.22|2.31  |2.26 |[706, 0, 537]                                                           |[582, 0, 514]                                                           |[582, 0, 514]                                                           |[582, 0, 514]                                                           |
|Pipeline |CNN     |4             |2               |6.48 |3.31 |3.36  |3.43  |[99, 611, 534, 517]                                                     |[69, 514, 513, 514]                                                     |[69, 514, 513, 514]                                                     |[69, 514, 513, 514]                                                     |
|Pipeline |CNN     |5             |2               |7.95 |3.97 |4.0  |4.14  |[104, 87, 611, 534, 516]                                                |[79, 45, 514, 513, 513]                                                 |[79, 45, 514, 513, 513]                                                 |[79, 45, 514, 513, 513]                                                 |
|Pipeline |CNN     |6             |2               |9.27  |5.427 |5.37 |5.447  |[104, 93, 0, 611, 534, 516]                                             |[79, 51, 0, 514, 513, 513]                                              |[79, 51, 0, 514, 513, 513]                                              |[79, 51, 0, 514, 513, 513]                                              |
|Pipeline |CNN     |7             |2               |9.72  |5.97 |5.63  |5.86   |[79, 110, 0, 611, 534, 0, 516]                                          |[54, 68, 0, 514, 513, 0, 513]                                           |[54, 68, 0, 514, 513, 0, 513]                                           |[54, 68, 0, 514, 513, 0, 513]                                           |
|Pipeline |FFNET   |1             |2               |1.37|0.67|0.7 |0.71 |[830]                                                                   |[668]                                                                   |[668]                                                                   |[668]                                                                   |
|Pipeline |FFNET   |2             |2               |2.54 |1.4|1.37  |1.45  |[681, 517]                                                              |[522, 514]                                                              |[522, 514]                                                              |[522, 514]                                                              |
|Pipeline |VisionTransformer|2             |2               |2270.45 |2269.89|2270.93  |2271.06 |[4593434, 4644371]                                                      |[3979519, 3958031]                                                      |[3979519, 3958031]                                                      |[3979519, 3958031]                                                      |
|Pipeline |VisionTransformer|3             |2               |2015.97|2014.98|2015.59 |2016.3 |[2970001, 3472961, 3062524]                                             |[2574197, 3021793, 2635734]                                             |[2574197, 3021793, 2635734]                                             |[2574197, 3021793, 2635734]                                             |
|Pipeline |VisionTransformer|4             |2               |1862.57|1861.44 |1862.19 |1862.48   |[2413209, 2563197, 2521879, 2436588]                                    |[2119623, 2228904, 2145561, 2112281]                                    |[2119623, 2228904, 2145561, 2112281]                                    |[2119623, 2228904, 2145561, 2112281]                                    |
|Pipeline |VisionTransformer|5             |2               |1788.47|1786.39|1786.99 |1787.25 |[1935877, 2230125, 2003022, 2197198, 1958722]                           |[1706017, 1918617, 1732474, 1925785, 1657517]                           |[1706017, 1918617, 1732474, 1925785, 1657517]                           |[1706017, 1918617, 1732474, 1925785, 1657517]                           |
|Pipeline |VisionTransformer|6             |2               |1718.44|1715.75|1716.21 |1716.23  |[1670025, 1965041, 1765997, 1938513, 1765997, 1693225]                  |[1478507, 1691115, 1546192, 1664987, 1546192, 1430453]                  |[1478567, 1691115, 1546192, 1664987, 1546192, 1430453]                  |[1478567, 1691115, 1546192, 1664987, 1546192, 1430453]                  |
|Pipeline |VisionTransformer|7             |2               |1680.24|1676.8 |1676.74 |1676.7 |[1616444, 1470947, 1618210, 1705620, 1470947, 1671737, 1454147]         |[1437800, 1277582, 1436969, 1498672, 1277582, 1477999, 1229807]         |[1437800, 1277582, 1436969, 1498672, 1277582, 1477999, 1229807]         |[1437800, 1277582, 1436969, 1498672, 1277582, 1477999, 1229807]         |
|Pipeline |VisionTransformer|8             |2               |1635.16|1631.67|1632.07   |1632.32  |[1350652, 1446284, 1445966, 1492565, 1485961, 1431457, 1377890, 1366146]|[1210541, 1278218, 1277900, 1312002, 1276935, 1209459, 1209459, 1195574]|[1210541, 1278218, 1277900, 1312002, 1276935, 1209459, 1209459, 1195574]|[1210541, 1278218, 1277900, 1312002, 1276935, 1209459, 1209459, 1195574]|


### Result analysis
Firstly, we notice that some results are missing, for example for FFNET we only have results on 1 or 2 GPUs etc... It's simply that when an error occurs, the result is not stored in the benchmark. But there are 4 possible types of error: 

1. If the first/last GPU in the chain has only one layer, it cannot be executed.
2. One of the GPUs has 0 layers.
3. Cuda Out of Memory, at least 1 of the GPUs can't handle the amount of layers and data given to it.
4. Finally, an error occurs during training.

If none of these errors occur, the results are stored.

So, based on this, we can see right away that no input is available with the torch API for the VisionTransformer, simply because it doesn't fit on a single GPU. As a result, the pipeline tool allows the model to be separated on multiple GPUs. Another point to note is that the tool slows down execution time anyway, due to the added communication between GPUs. So it can't be used routinely, and should only be used in really useful cases, i.e. when the model can't fit on a single GPU.

## Improvements
1. Although repartition is currently performed, it is unnecessary when the model fits within a single GPU. The process should automatically avoid splitting when feasible, requiring an initial run on the largest GPU and an error-handling mechanism.
2. Replace the rudimentary repartition method with a more efficient approach, such as employing a dichotomous search.
3. Actually, the tool is searching for the best memory balancing between GPU. But after some execution time analysis this solution is not the best concerning execution time. One improvement should be to instead of search for the best memory balancing is to search for the best execution time. To put this solution in place : 
    1. Change the analysis return by the script `evaluate_mem.py` to return time and not memory balancing.
    2. Find a way to preprocess and create all potential best repartition to avoid testing all possibility that could exponetial on process time depending on the number of layers.
    3. Change the behaviour to test all pre-calculated possibility and not stop and keep the fastest one.

    Here the time analysis made, in italic are the chosen repartition and in bold the minimal execution time:

    | **Model**         | **Nb GPU** | **Repartition**                  | **Epoch 1** | **Epoch 2** | **Epoch 3** | **Minimal Epoch time** |
    | ----------------- | ---------- | -------------------------------- | ----------- | ----------- | ----------- | ---------------------- |
    | CNN               | 2          | [7, 7]                           | 3.85        | 1.84        | **1.79**    |                        |
    | CNN               | 2          | [8, 6]                           | 3.81        | 1.91        | 1.85        |                        |
    | *CNN*             | *2*        | *[9, 5]*                         | *4.02*      | *1.89*      | *1.84*      | 1.79                   |
    |                   |            |                                  |             |             |             |                        |
    | CNN               | 3          | [5, 5, 4]                        | 5.08        | 2.49        | 2.55        |                        |
    | CNN               | 3          | [6, 4, 4]                        | 5.05        | 2.65        | 2.66        |                        |
    | CNN               | 3          | [7, 3, 4]                        | 4.95        | 2.62        | 2.51        |                        |
    | CNN               | 3          | [8, 2, 4]                        | 5.13        | 2.54        | 2.62        |                        |
    | *CNN*             | *3*        | *[9, 1, 4]*                      | *4.20*      | *2.21*      | **2.21**    | 2.21                   |
    |                   |            |                                  |             |             |             |                        |
    | CNN               | 4          | [4, 4, 3, 3]                     | 6.66        | 3.20        | 3.32        |                        |
    | CNN               | 4          | [4, 5, 2, 3]                     | 6.65        | 3.36        | 3.40        |                        |
    | CNN               | 4          | [5, 4, 2, 3]                     | 6.20        | 3.32        | 3.25        |                        |
    | CNN               | 4          | [6, 3, 2, 3]                     | 6.14        | 3.21        | 3.16        |                        |
    | CNN               | 4          | [7, 2, 2, 3]                     | 6.07        | 3.23        | 3.28        |                        |
    | *CNN*             | *4*        | *[8, 1, 2, 3]*                   | *6.08*      | *3.31*      | *3.35*      |                        |
    | CNN               | 4          | [9, 1, 1, 3]                     | 5.39        | 2.88        | **2.88**    | 2.88                   |
    |                   |            |                                  |             |             |             |                        |
    | CNN               | 5          | [3, 3, 3, 3, 2]                  | 7.96        | 3.96        | 3.85        |                        |
    | CNN               | 5          | [3, 4, 2, 3, 2]                  | 7.81        | 3.87        | 3.73        |                        |
    | CNN               | 5          | [3, 5, 1, 3, 2]                  | 7.86        | 3.85        | 4.05        |                        |
    | CNN               | 5          | [3, 6, 1, 2, 2]                  | 7.05        | 3.61        | **3.53**    |                        |
    | *CNN*             | *5*        | *[3, 5, 2, 2, 2]*                | *7.87*      | *3.81*      | *3.91*      | 3.53                   |
    |                   |            |                                  |             |             |             |                        |
    | *CNN*             | *6*        | *[3, 3, 2, 2, 2, 2]*             | *8.95*      | *4.98*      | *4.79*      |                        |
    | CNN               | 6          | [3, 3, 3, 1, 2, 2]               | 8.10        | **4.07**    | 4.19        | 4.07                   |
    |                   |            |                                  |             |             |             |                        |
    | CNN               | 7          | [2, 2, 2, 2, 2, 2, 2]            | 8.55        | 4.60        | 4.64        |                        |
    | CNN               | 7          | [2, 3, 2, 2, 1, 2, 2]            | 9.59        | 5.69        | 5.53        |                        |
    | *CNN*             | *7*        | *[2, 3, 3, 1, 1, 2, 2]*          | *9.26*      | *5.71*      | *5.69*      |                        |
    | CNN               | 7          | [2, 3, 4, 1, 1, 1, 2]            | 8.44        | 4.56        | **4.42**    | 4.42                   |
    |                   |            |                                  |             |             |             |                        |
    | *FFNET*           | *2*        | *[3, 2]*                         | *2.39*      | *1.32*      | **1.27**    |                        |
    | FFNET             | 2          | [2, 3]                           | 2.41        | 1.39        | 1.34        | 1.27                   |
    |                   |            |                                  |             |             |             |                        |
    | *VisionTransformer* | *2*      | *[184, 184]*                     | *471.70*    | *470.42*    | *470.53*    |                        |
    |                   |            |                                  |             |             |             |                        |
    | *VisionTransformer* | *3*      | *[123, 123, 122]*                | *418.28*    | *416.19*    | *416.29*    |                        |
    |                   |            |                                  |             |             |             |                        |
    | *VisionTransformer* | *4*      | *[92, 92, 92, 92]*               | *385.19*    | *382.62*    | *383.39*    |                        |
    |                   |            |                                  |             |             |             |                        |
    | *VisionTransformer* | *5*      | *[74, 74, 74, 73, 73]*           | *370.09*    | *367.58*    | *367.66*    |                        |
    |                   |            |                                  |             |             |             |                        |
    | VisionTransformer | 6          | [62, 62, 61, 61, 61, 61]         | 356.13      | **353.30**  | 353.54      |                        |
    | *VisionTransformer* | *6*      | *[63, 62, 61, 60, 61, 61]*       | *357.41*    | *354.58*    | *354.84*    | 353.30                 |
    |                   |            |                                  |             |             |             |                        |
    | VisionTransformer | 7          | [53, 53, 53, 53, 52, 52, 52]     | 347.54      | **345.12**  | 345.18      |                        |
    | VisionTransformer | 7          | [54, 53, 52, 53, 52, 52, 52]     | 351.04      | 347.78      | 347.89      |                        |
    | VisionTransformer | 7          | [55, 52, 52, 53, 52, 52, 52]     | 349.60      | 346.24      | 346.29      |                        |
    | VisionTransformer | 7          | [56, 52, 51, 53, 52, 52, 52]     | 349.48      | 346.58      | 346.45      |                        |
    | VisionTransformer | 7          | [57, 52, 50, 53, 52, 52, 52]     | 349.51      | 346.42      | 346.55      |                        |
    | VisionTransformer | 7          | [58, 52, 49, 53, 52, 52, 52]     | 348.30      | 345.28      | 345.35      |                        |
    | *VisionTransformer* | *7*      | *[59, 52, 49, 53, 52, 51, 52]*   | *348.69*    | *345.15*    | *345.28*    | 345.12                 |
    |                   |            |                                  |             |             |             |                        |
    | VisionTransformer | 8          | [46, 46, 46, 46, 46, 46, 46, 46] | 342.10      | 338.47      | 338.73      |                        |
    | VisionTransformer | 8          | [47, 45, 46, 46, 46, 46, 46, 46] | 342.17      | 338.51      | 338.44      |                        |
    | *VisionTransformer* | *8*      | *[48, 44, 46, 46, 46, 46, 46, 46]* | *339.99*  | *336.45*    | **336.44**  | 336.44                 |

## Known issue
1. The pipeline of Persformers in more than 2 GPU have backward process problem for an unknow reason and no error is throw.
2. Actually when a CUDA error OOM is throw we admit that the envrionement don't have enough GPU. In the futur we will implement logic test to see if yes or not the model can be split on the desired config before telling that it is impossible.