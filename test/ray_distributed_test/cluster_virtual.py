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

def test_get_node_ip_and_free_port_does_not_start_with_zero():
    # This test covers a case where the hostname was an integer like "255"
    # and socket returned an ip address equivalent to this hostname, i.e., "0.0.0.255".
    # It's not possible to mock the way the hostname is actually set on other platforms,
    # so we leave this test so we can ask users to run on their environment if needed.

    node_ip, _ = ray.get(
        _get_node_ip_and_free_port.options(
            runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
        ).remote()
    )
    assert not node_ip.startswith("0."), "Node IP should not start with 0.*.*.*"

def create_cluster():
    init_ray()
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[8], use_gpus=True)
    print(cluster)
    cluster._init_placement_groups()
    print(cluster.world_size(), cluster.node_count())
    # cluster.get_placement_groups()
    cluster.shutdown()

    ray.shutdown()




    

if __name__ == '__main__':
    create_cluster()
    test_get_node_ip_and_free_port_does_not_start_with_zero()