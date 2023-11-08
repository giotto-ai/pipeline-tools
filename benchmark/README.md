# Benchmark of Pipeline Tool

## In local
The benchmark of the pipeline tool allows for checking the proper operation of the module in the environment, as well as verifying its execution speed and memory distribution on various GPUs. To obtain meaningful results, benchmarking is conducted on three different models:

- A FFNET
- A CNN
- A Vision Transformer

It involves running a complete model parsing, assessing its distribution, and conducting a training trial to verify the time and resources involved under real-world conditions.

To initiate the benchmark, a script is provided: [benchmark.sh](benchmark.sh), which takes as its sole parameter the desired maximum number of GPUs. It will then run a trial for each model using the standard torch API to establish a baseline and subsequently run the pipeline tool for the three models on 1 to N GPUs, where N is the maximum specified in the script.

The results will be stored in a text file in the following format:

```txt
Framework;Model;Number of GPUs;Number of Chunks;Time 1 [s];Time 2 [s];Time 3 [s];Time 4 [s];Alloc 1 [MB];Alloc 2 [MB];Alloc 3 [MB];Alloc 4 [MB]
API torch;CNN;1;0;2.3584954738616943;0.4620656967163086;0.45350098609924316;0.446514368057251;[747];[625];[625];[625]
[...]
Pipeline;CNN;1;2;2.887887954711914;0.9743552207946777;0.943678617477417;0.9702959060668945;[774];[628];[628];[628]
[...]
```
## On GKE 
To configure the command of this section, populate the variables below:
```bash
PROJECT_ID=""
CLUSTER_ZONE=""
BUCKET=""
SA_KUBE=""
ARTIFACT_REGISTRY=""
IMAGE_NAME="pipeline-benchmark:latest"
IMAGE_FULLPATH="${CLUSTER_ZONE}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY}/${IMAGE_NAME}"
```

### Build Benchmark image
The Docker image is build on nvidia/cuda runtime image. 

Execute this steps from the root of the project.
```bash
cp benchmark/Dokerfile .
docker builder build -t ${IMAGE_FULLPATH} .
docker push ${IMAGE_FULLPATH}
rm -f Dockerfile
```

### Run deployment on GKE

To simplify the creation of a Kubernetes pod, a script is provided, `gen_pod.py.` This script will enable the population of a template `pod_template.yml` and create a pod ready to be applied to a Kubernetes cluster.

Here is how to use `gen_pod.py`:
```bash
python3 gen_pod.py -i $IMAGE_FULLPATH -b $BUCKET -k $SA_KUBE -c 4 -g a100
```
Important thing to know is that the number of GPU set with -c will be passed to the script [benchmark.sh](benchmark.sh)
And then apply the pod to the kubernets with : 
```bash
kubectl apply -f pod-a100-4.yml
```
