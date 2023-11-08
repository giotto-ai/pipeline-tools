import argparse
import enum
import string

class GPUs(enum.Enum):
    a100 = enum.auto()
    v100 = enum.auto()
    t4 = enum.auto()

    def __str__(self) -> str:
        return self.name
    
    @staticmethod
    def from_string(s):
        try:
            return GPUs[s]
        except KeyError:
            raise ValueError()
        
    def fullname(self) -> str:
        if self is GPUs.a100:
            return "nvidia-tesla-a100"
        elif self is GPUs.v100:
            return "nvidia-tesla-v100"
        elif self is GPUs.t4:
            return "nvidia-tesla-t4"
        else:
            raise Exception(f"Fullname missing for {self.name}")

def run(args):
    values = {
        "image": args.image,
        "bucket": args.bucket,
        "ksa": args.ksa,
        "gpu_count": args.gpu_count,
        "gpu_model": args.gpu_model.fullname(),
    }
    
    with open("pod_template.yml", "r") as f:
        ymlt = string.Template(f.read())
    
    ymlv = ymlt.substitute(values)
    filename = f"pod-{args.gpu_model}-{args.gpu_count}.yml"
    with open(filename, "w") as f:
        f.write(ymlv)

    print(f"kubectl apply -f {filename}")

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", required=True, help="Container image")
parser.add_argument("-b", "--bucket", required=True, help="Storage bucket")
parser.add_argument("-k", "--ksa", required=True, help="Kubernetes Service Account")
parser.add_argument("-c", "--gpu-count", required=True, type=int, help="GPU count")
parser.add_argument("-g", "--gpu-model", required=True, type=GPUs.from_string, choices=[x for x in GPUs], help="GPU model")

args = parser.parse_args()
run(args)
