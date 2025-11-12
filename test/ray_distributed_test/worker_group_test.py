
import os
import sys

import pytest
import ray

from omini_vex.ray_distributed.named_sharding import NamedSharding


from omini_vex.ray_distributed.virtual_cluster import (
    PY_EXECUTABLES,
    RayVirtualCluster,
    ResourceInsufficientError,
    _get_node_ip_and_free_port,
    init_ray
)
import os
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
import ray
import sys 
from unittest.mock import MagicMock, patch




@ray.remote
class MyTestActor:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.configured_gpus_in_init = kwargs.get("configured_gpus", "not_set")
        self.bundle_indices_seen_in_init = kwargs.get(
            "bundle_indices_seen_in_init", "not_set"
        )
        self.env_vars = dict(os.environ)
        self.pid = os.getpid()
        self.stored_data = None
        self.stored_args = None
        self.stored_kwargs = None
        self.call_count = 0

    def get_pid(self):
        return self.pid

    def get_init_args_kwargs(self):
        return self.init_args, self.init_kwargs

    def get_env_var(self, var_name):
        return self.env_vars.get(var_name)

    def echo(self, x):
        return f"Actor {self.pid} echoes: {x}"

    def get_rank_world_size_node_rank_local_rank(self):
        return (
            self.env_vars.get("RANK"),
            self.env_vars.get("WORLD_SIZE"),
            self.env_vars.get("NODE_RANK"),
            self.env_vars.get("LOCAL_RANK"),
        )

    def get_master_addr_port(self):
        return self.env_vars.get("MASTER_ADDR"), self.env_vars.get("MASTER_PORT")

    def check_configured_worker_effect(self):
        return (
            self.configured_gpus_in_init,
            self.bundle_indices_seen_in_init,
            self.env_vars.get("CONFIGURED_WORKER_CALLED"),
        )

    def get_actual_python_executable_path(self):
        return sys.executable

    def record_call(self, data=None, *args, **kwargs):
        self.stored_data = data
        self.stored_args = args
        self.stored_kwargs = kwargs
        self.call_count += 1
        return f"Actor {self.pid} called with data: {data}, args: {args}, kwargs: {kwargs}, call_count: {self.call_count}, my_rank: {self.env_vars.get('RANK')}"

    def get_recorded_data(self):
        return self.stored_data, self.stored_args, self.stored_kwargs, self.call_count

    def reset_call_records(self):
        self.stored_data = None
        self.stored_args = None
        self.stored_kwargs = None
        self.call_count = 0

    @staticmethod
    def configure_worker(num_gpus, bundle_indices):
        init_kwargs_update = {
            "configured_gpus": num_gpus,
            "bundle_indices_seen_in_init": bundle_indices is not None,
        }
        resources = {"num_gpus": num_gpus}
        env_vars_update = {"CONFIGURED_WORKER_CALLED": "1"}
        return resources, env_vars_update, init_kwargs_update


def worker_group_1d_sharding(virtual_cluster):
    MY_TEST_ACTOR_FQN = f"{MyTestActor.__module__}.MyTestActor"
    print(MY_TEST_ACTOR_FQN)
    # actor_fqn = MY_TEST_ACTOR_FQN
    # builder = RayWorkerBuilder(actor_fqn)
    # # virtual_cluster has 1 node, 2 bundles. Sharding will be across these 2 bundles.
    # sharding = NamedSharding(layout=[0, 1], names=["data"])
    # worker_group = RayWorkerGroup(
    #     cluster=virtual_cluster,
    #     remote_worker_builder=builder,
    #     workers_per_node=None,  # Should create 2 workers, one per bundle
    #     sharding_annotations=sharding,
    # )
    # yield worker_group
    # worker_group.shutdown(force=True)

if __name__ == '__main__':
    worker_group_1d_sharding(None)