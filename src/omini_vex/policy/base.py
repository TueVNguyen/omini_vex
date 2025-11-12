from typing import Any, Literal, NotRequired, TypedDict, Union


# This config for policy model 
# We will support sequence_parallel first

class DTensorConfig(TypedDict):
    enabled: Literal[True]
    env_vars: NotRequired[dict[str, str] | None]

    cpu_offload: bool
    sequence_parallel: bool
    activation_checkpointing: bool
    tensor_parallel_size: int
    custom_parallel_plan: str | None


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer_path: str
    dtensor_cfg: DTensorConfig