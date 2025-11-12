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


def main():
    print("\n" + "="*80)
    print("EXPLORING RAYWORKERGROUP")
    print("="*80)

    # Initialize Ray
    if not ray.is_initialized():
        print("\n📍 Step 1: Initializing Ray...")
        ray.init(log_to_driver=False, include_dashboard=False, num_cpus=8, num_gpus=0)
        print("   ✓ Ray initialized with 8 CPUs, 0 GPUs")

    # Register actor
    actor_fqn = f"{MyTestActor.__module__}.MyTestActor"
    ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.SYSTEM
    print(f"\n📍 Step 2: Registered actor")
    print(f"   Actor FQN: {actor_fqn}")
    print(f"   Python executable: {PY_EXECUTABLES.SYSTEM}")

    # Create virtual cluster
    print(f"\n📍 Step 3: Creating virtual cluster")
    print(f"   Configuration:")
    print(f"   - bundle_ct_per_node_list=[2, 2]  # 2 nodes with 2 bundles each")
    print(f"   - use_gpus=False                   # CPU only")

    cluster = RayVirtualCluster(bundle_ct_per_node_list=[2, 2], use_gpus=False)

    print(f"   ✓ Virtual cluster created:")
    print(f"     - Total bundles: {cluster.world_size()}")
    print(f"     - Node count: {cluster.node_count()}")
    master_addr, master_port = cluster.get_master_address_and_port()
    print(f"     - Master address: {master_addr}:{master_port}")

    # Create 2D sharding
    print(f"\n📍 Step 4: Creating 2D sharding annotation")
    print(f"   Layout: [[0, 1], [2, 3]]")
    print(f"   Names: ['dp', 'tp']")
    print(f"   This means:")
    print(f"   - dp (data parallel) axis has size 2")
    print(f"   - tp (tensor parallel) axis has size 2")
    print(f"   - Worker 0 is at (dp=0, tp=0)")
    print(f"   - Worker 1 is at (dp=0, tp=1)")
    print(f"   - Worker 2 is at (dp=1, tp=0)")
    print(f"   - Worker 3 is at (dp=1, tp=1)")

    sharding = NamedSharding(layout=[[0, 1], [2, 3]], names=["dp", "tp"])
    print(f"   ✓ Sharding created:")
    print(f"     - Shape: {sharding.shape}")
    print(f"     - Layout array:\n{sharding.layout}")

    # Create worker builder
    print(f"\n📍 Step 5: Creating worker builder")
    builder = RayWorkerBuilder(actor_fqn)
    print(f"   ✓ Builder created for {actor_fqn}")

    # Create custom environment variables
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

    print(f"   ✓ Worker group created with {len(worker_group.workers)} workers")

    # Explore worker metadata
    print(f"\n📍 Step 7: Exploring worker metadata")
    for i, metadata in enumerate(worker_group.worker_metadata):
        coords = sharding.get_worker_coords(i)
        print(f"   Worker {i}:")
        print(f"     - Coordinates: {coords}")
        print(f"     - Node index: {metadata['node_idx']}")
        print(f"     - Global rank: {metadata['global_rank']}")
        print(f"     - Local rank: {metadata['local_rank']}")
        print(f"     - DP shard index: {metadata['dp_shard_idx']}")

    # Test remote method calls
    print(f"\n📍 Step 8: Testing remote method calls")
    print(f"   Calling echo() on each worker...")

    futures = [w.echo.remote(f"Message from worker {i}") for i, w in enumerate(worker_group.workers)]
    results = ray.get(futures)

    for i, result in enumerate(results):
        print(f"   Worker {i}: {result}")

    # Check environment variables
    print(f"\n📍 Step 9: Checking environment variables in workers")
    for i, worker in enumerate(worker_group.workers):
        rank, ws, node_rank, local_rank = ray.get(
            worker.get_rank_world_size_node_rank_local_rank.remote()
        )
        custom_var = ray.get(worker.get_env_var.remote("MY_CUSTOM_VAR"))

        print(f"   Worker {i}:")
        print(f"     - RANK={rank}, WORLD_SIZE={ws}")
        print(f"     - NODE_RANK={node_rank}, LOCAL_RANK={local_rank}")
        print(f"     - MY_CUSTOM_VAR={custom_var}")

    # Test sharded data distribution
    print(f"\n📍 Step 10: Testing sharded data distribution")
    print(f"   Sharding data along 'dp' axis, replicating on 'tp' axis")

    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    data_dp0 = {"dp_id": 0, "message": "Data for DP group 0"}
    data_dp1 = {"dp_id": 1, "message": "Data for DP group 1"}
    sharded_data = [data_dp0, data_dp1]

    print(f"   Input data: {sharded_data}")

    future_bundle = worker_group.run_all_workers_sharded_data(
        "record_call",
        data=sharded_data,
        in_sharded_axes=["dp"],      # Shard along dp
        replicate_on_axes=["tp"],    # Replicate on tp
    )

    results = worker_group.get_all_worker_results(future_bundle)

    print(f"   ✓ Method called on {len(future_bundle.called_workers)} workers")
    print(f"   ✓ Got {len(results)} results back")

    print(f"\n   Data received by each worker:")
    for i in range(len(worker_group.workers)):
        data, _, _, count = ray.get(worker_group.workers[i].get_recorded_data.remote())
        coords = sharding.get_worker_coords(i)
        print(f"     Worker {i} {coords}:")
        print(f"       - Call count: {count}")
        print(f"       - Data received: {data}")

    # Test filtering by axis
    print(f"\n📍 Step 11: Testing axis filtering (only run on tp=0)")
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    test_data = {"message": "Only for tp=0 workers"}
    futures = worker_group.run_all_workers_single_data(
        "record_call",
        data=test_data,
        run_rank_0_only_axes=["tp"],  # Only run on workers where tp=0
    )

    print(f"   ✓ Called {len(futures)} workers (only those with tp=0)")

    print(f"\n   Which workers were called:")
    for i in range(len(worker_group.workers)):
        _, _, _, count = ray.get(worker_group.workers[i].get_recorded_data.remote())
        coords = sharding.get_worker_coords(i)
        status = "✓ CALLED" if count > 0 else "✗ NOT CALLED"
        print(f"     Worker {i} {coords}: {status}")

    # Cleanup
    print(f"\n📍 Step 12: Cleanup")
    worker_group.shutdown(force=True)
    cluster.shutdown()
    ray.shutdown()
    print(f"   ✓ All resources cleaned up")

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE! 🎉")
    print("="*80 + "\n")

    print("💡 Tips for exploration:")
    print("  - Add more print() statements wherever you want")
    print("  - Use ray.get() to get values from workers")
    print("  - Check the worker_group.workers list")
    print("  - Explore sharding.get_worker_coords(worker_idx)")
    print("  - Try different sharding configurations")
    print("")


if __name__ == "__main__":
    main()
