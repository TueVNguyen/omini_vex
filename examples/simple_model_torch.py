#!/usr/bin/env python3
"""Simple exploration script to understand RayWorkerGroup with lots of print statements."""

import os
import sys
# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray

from omini_vex.ray_distributed.named_sharding import NamedSharding
from omini_vex.ray_distributed.ray_actor_enviroment_registry import ACTOR_ENVIRONMENT_REGISTRY
from omini_vex.ray_distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from omini_vex.ray_distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup

# Import the test actor from the test module
from test.ray_distributed_test.test_actors import MyTestActor

from omini_vex.policy.simple_torch_dtensor import SimpleDtensorModel


def main():
    if not ray.is_initialized():
        print("\n📍 Step 1: Initializing Ray...")
        ray.init(log_to_driver=False, include_dashboard=False, num_cpus=32, num_gpus=8)
        print("   ✓ Ray initialized with 32 CPUs, 8 GPUs")


    actor_fqn = f"{SimpleDtensorModel.__module__}.SimpleDtensorModel"
    ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.SYSTEM
    print(f"\n📍 Step 2: Registered actor")
    print(f"   Actor FQN: {actor_fqn}")
    print(f"   Python executable: {PY_EXECUTABLES.SYSTEM}")

    cluster = RayVirtualCluster(bundle_ct_per_node_list=[4], use_gpus=True)

    master_addr, master_port = cluster.get_master_address_and_port()

    sharding = NamedSharding(layout=[0, 1, 2, 3], names=["dp"])

    builder = RayWorkerBuilder(actor_fqn)

    custom_env = {
        "MY_CUSTOM_VAR": "hello_world",
        "DEBUG": "true",
    }
    print(f"\n📍 Step 6: Creating worker group")
    print(f"   Custom env vars: {custom_env}")

    worker_group = RayWorkerGroup(
        cluster=cluster,
        remote_worker_builder=builder,
        workers_per_node=None,
        sharding_annotations=sharding,
        env_vars=custom_env,
    )
    futures = [w.echo.remote(f"Message from worker {i}") for i, w in enumerate(worker_group.workers)]
    results = ray.get(futures)

    for i, result in enumerate(results):
        print(f"   Worker {i}: {result}")


    print(f"   ✓ Worker group created with {len(worker_group.workers)} workers")
    ray.shutdown()

if __name__ == '__main__':
    main()