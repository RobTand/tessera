"""Actual rank, device and host binding for multiworker full-engine observers."""
import fcntl
import os
import socket
import struct


def actual_worker_identity(plan, configured_world, rank, local_rank, visible_uuids):
    """Bind one actual vLLM rank to its local physical device before CUDA."""
    declared = plan["identity"]
    if (type(configured_world) is not int or configured_world not in (1, 2)
            or plan.get("world_size", declared.get("world_size")) != configured_world):
        raise ValueError("actual worker world disagrees with the TP1/TP2 plan")
    if (type(rank) is not int or rank not in range(configured_world)
            or type(local_rank) is not int or local_rank not in range(len(visible_uuids))):
        raise ValueError("actual worker rank or local device is outside the configured world")
    uuid = visible_uuids[local_rank]
    if type(uuid) is not str or not uuid:
        raise ValueError("actual worker GPU UUID is unavailable")
    return dict(declared, rank=rank, world_size=configured_world,
                device_id=local_rank, device_uuid=uuid)


def actual_host_ip():
    """Require vLLM's configured host IP to belong to this worker's network.

    The GLM TP2 launch uses the host network. ``VLLM_HOST_IP`` is a configured
    transport input; the local interface census is the independent observation
    that binds that spelling to the process's actual host network.
    """
    configured = os.environ.get("VLLM_HOST_IP")
    if not configured:
        raise ValueError("TP2 worker has no configured VLLM_HOST_IP")
    try:
        socket.inet_aton(configured)
    except OSError as exc:
        raise ValueError("TP2 VLLM_HOST_IP is not an IPv4 address") from exc
    observed = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            try:
                record = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))
            except OSError:
                continue
            observed.append({"interface": name, "ipv4": socket.inet_ntoa(record[20:24])})
    matches = [row for row in observed if row["ipv4"] == configured]
    if len(matches) != 1:
        raise ValueError("configured VLLM_HOST_IP is not on one actual worker interface")
    return {"ip": configured, "interface": matches[0]["interface"],
            "source": "VLLM_HOST_IP equal to the worker's own IPv4 interface address"}
