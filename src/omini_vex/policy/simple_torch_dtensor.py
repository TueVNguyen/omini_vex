import ray

from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.tensor import DTensor, Shard
import torch
import os 

class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,7)
    
    def forward(self, inputs):
        return self.linear(inputs)

@ray.remote
class SimpleDtensorModel:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"
    


    def __init__(
        self, *args, **kwargs
    ):
        print(args, kwargs)
        torch._inductor.config.autotune_local_cache = False
        torch.distributed.init_process_group(backend="nccl")
        self.rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # torch==2.8 uses LOCAL_RANK to set the device here (https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/device_mesh.py#L500),
        # but CUDA_VISIBLE_DEVICES is set to only 1 gpu, so we need to temporarily set LOCAL_RANK to 0.
        # TODO: consider changing the default LOCAL_RANK set in worker_groups.py

        prev_local_rank = os.environ["LOCAL_RANK"]
        os.environ["LOCAL_RANK"] = "0"
        dp_replicate_size = world_size
        dp_shard_size = 1
        cp_size = 1
        tp_size = 1
        dp_size = dp_replicate_size
        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (dp_replicate_size, dp_shard_size, cp_size, tp_size),
            mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"),
        )
        os.environ["LOCAL_RANK"] = prev_local_rank

        device_mesh[("dp_replicate", "dp_shard")]._flatten(mesh_dim_name="dp")
        # Flatten dp_shard + cp for FSDP2 sharding
        device_mesh[("dp_shard", "cp")]._flatten(mesh_dim_name="dp_shard_cp")
        # Flatten dp_replicate + dp_shard + cp for gradient operations
        device_mesh[("dp_replicate", "dp_shard", "cp")]._flatten(mesh_dim_name="dp_cp")

        # Store mesh references for backward compatibility
        self.dp_cp_mesh = device_mesh["dp_cp"]
        self.dp_mesh = device_mesh["dp"]
        self.tp_mesh = device_mesh["tp"]
        self.cp_mesh = device_mesh["cp"]

        self.dp_size = dp_size
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.device_mesh = device_mesh

    
        self.model = FakeModel()
        print(self.model)
    
    def echo(self, x):
        return str(self.model) + str(self)