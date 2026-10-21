"""Two-node manual smoke for the public elastic Mooncake EP dispatcher.

Run from the SGLang repo root in myenv, with the Mooncake EP native package
installed and one visible GPU per node. Use the same reachable MASTER_ADDR and
these distinct free ports on both nodes:

    export MASTER_ADDR=<node-A-IP> MASTER_PORT=29500
    export SGLANG_TEST_DIST_PORT=29501 SGLANG_TCP_STORE_PORT=29502
    export SGLANG_TEST_MOONCAKE_IB_DEVICE=mlx5_1
    export SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8

Node A:
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nnodes=2 \\
        --nproc-per-node=1 --node-rank=0 --master-addr=$MASTER_ADDR \\
        --master-port=$MASTER_PORT test/manual/ep/test_mooncake_dispatcher.py

Node B runs the same command with --node-rank=1. This test uses real
ServerArgs, RuntimeContext publication, bootstrap, process groups, and Mooncake PG. It tests
capacity 4 with two active ranks, FP8-wire dispatch with BF16 inputs, identity
expert output, and both event-wait and receive-hook paths. Inputs are constant per
128-wide scale block, so standard FP8 scale/dequant has an exact BF16 reference. Public SGLang
Mooncake dispatch selects FP8; this is not a BF16-wire test (P1 native suite
covers that). No model, TBO, fault, Graph, or scale-up is tested.
"""

import os
import socket
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_moe_ep_group,
)
from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.mooncake import (
    EPBuffer,
    MooncakeEPDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.runtime_context import SpawnRanks, get_parallel, get_resources, publish
from sglang.srt.server_args import prepare_server_args

CAPACITY, ACTIVE, EPR, TOKENS, HIDDEN, MAX_TOKENS = 4, 2, 2, 4, 2048, 8


def _server_args(rank: int):
    argv = f"""--model-path dummy --device cuda --trust-remote-code
        --tp {ACTIVE} --ep-size {ACTIVE} --attn-dp-size {ACTIVE}
        --nnodes 2 --node-rank {os.environ.get("GROUP_RANK", rank)}
        --dist-init-addr {os.environ["MASTER_ADDR"]}:{os.environ["SGLANG_TEST_DIST_PORT"]}
        --moe-a2a-backend mooncake --elastic-ep-backend mooncake
        --mooncake-ib-device {os.environ["SGLANG_TEST_MOONCAKE_IB_DEVICE"]} --deepep-mode low_latency
        --max-ep-size {CAPACITY} --elastic-ep-initial-size {ACTIVE}
        --moe-dense-tp-size 1 --enable-dp-lm-head --disable-custom-all-reduce
        --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled"""
    return prepare_server_args(argv.split())


def _identity_output(output):
    recv, scales, counts = (
        output.hidden_states,
        output.hidden_states_scale,
        output.masked_m,
    )
    assert recv.dtype == torch.float8_e4m3fn, recv.dtype
    assert scales is not None and scales.shape == (*recv.shape[:2], HIDDEN // 128)
    assert counts.dtype == torch.int32 and counts.shape == (EPR,)
    assert counts.tolist() == [TOKENS, 0], counts
    valid = torch.arange(recv.shape[1], device=recv.device)[None, :] < counts[:, None]
    valid_scales = scales[valid]
    assert torch.isfinite(valid_scales).all() and (valid_scales > 0).all()
    dequant = (recv.float() * scales.repeat_interleave(128, dim=-1)).to(torch.bfloat16)
    return torch.where(valid[:, :, None], dequant, 0)


def _run_case(rank: int, group, return_recv_hook: bool) -> None:
    dispatcher = MooncakeEPDispatcher(
        group=group,
        router_topk=1,
        num_experts=ACTIVE * EPR,
        num_local_experts=EPR,
        hidden_size=HIDDEN,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.LOW_LATENCY,
        return_recv_hook=return_recv_hook,
    )
    set_is_extend_in_batch(False)
    rows = torch.arange(TOKENS, dtype=torch.float32, device="cuda")[:, None]
    blocks = torch.arange(HIDDEN // 128, dtype=torch.float32, device="cuda")[None, :]
    block_values = (1 + rank * 0.5 + rows * 0.25 + blocks * 0.125).to(torch.bfloat16)
    hidden = block_values.repeat_interleave(128, dim=1).contiguous()
    assert torch.count_nonzero(hidden).item() == hidden.numel()
    topk_ids = torch.full(
        (TOKENS, 1), (1 - rank) * EPR, dtype=torch.int32, device="cuda"
    )
    weights = torch.ones((TOKENS, 1), dtype=torch.float32, device="cuda")

    dispatched = dispatcher.dispatch(
        hidden, StandardTopKOutput(weights, topk_ids, None)
    )
    buffer = EPBuffer.get_existing_buffer()
    assert buffer.group_size == CAPACITY
    assert EPBuffer.get_dispatch_ep_size() == ACTIVE
    from mooncake import ep

    print(
        f"MOONCAKE-DISPATCHER-NATIVE: native={ep.backend_module.__file__} "
        f"runtime={type(buffer.runtime).__module__}.{type(buffer.runtime).__name__}",
        flush=True,
    )
    assert dispatched.expected_m == 3, dispatched.expected_m
    assert dispatched.hidden_states.shape == (EPR, ACTIVE * MAX_TOKENS, HIDDEN)
    combined = dispatcher.combine(
        (_identity_output(dispatched), dispatched.topk_ids, dispatched.topk_weights)
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(combined, hidden, atol=0, rtol=0)

    state, buffer = ElasticEPStateManager.instance(), EPBuffer.get_existing_buffer()
    native_mask = torch.empty_like(state.active_ranks)
    buffer.query_mask_buffer(native_mask)
    assert native_mask.tolist() == [0, 0, 1, 1], native_mask
    assert state.active_ranks.tolist() == [1, 1, 0, 0], state.active_ranks
    error = (combined.float() - hidden.float()).abs().max().item()
    print(
        f"MOONCAKE-DISPATCHER: PASS rank={rank} hook={return_recv_hook} "
        f"backend={dist.get_backend(group)} capacity={buffer.group_size} "
        f"phase={EPBuffer.get_dispatch_ep_size()} mask={native_mask.tolist()} "
        f"active={state.active_ranks.tolist()} max_error={error:.5f}",
        flush=True,
    )


def _print_native_maps(rank: int) -> None:
    tokens = (
        "mooncake",
        "ep.cpython",
        "sgl_kernel",
        "libtorch",
        "libc10",
        "libcudart",
        "libcuda",
        "libnvrtc",
        "libnvjitlink",
        "libnccl",
    )
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps_file:
            paths = sorted(
                {
                    path
                    for line in maps_file
                    if (path := line.rstrip().split(None, 5)[-1]).startswith("/")
                    and any(token in path.lower() for token in tokens)
                }
            )
    except OSError as exc:
        paths = [f"<maps-read-error:{type(exc).__name__}>"]
    print(
        f"MOONCAKE-DISPATCHER-MAPS rank={rank} pid={os.getpid()} "
        f"paths={';'.join(paths) or '<none>'}",
        flush=True,
    )


class TestMooncakeDispatcher(unittest.TestCase):
    def test_elastic_dispatch_combine(self):
        if "RANK" not in os.environ:
            self.skipTest("run this manual GPU test with torchrun on two nodes")
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        assert world_size == ACTIVE, world_size
        os.environ["SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = str(
            MAX_TOKENS
        )
        server_args = _server_args(rank)
        publish(server_args, role="test", ranks=SpawnRanks(rank, gpu_id=local_rank))
        try:
            from sglang.srt.distributed.bootstrap import init_parallel_runtime

            init_parallel_runtime(
                server_args=server_args,
                device="cuda",
                dist_port=int(os.environ["SGLANG_TEST_DIST_PORT"]),
            )
            store = get_global_tcp_store()
            assert store is not None
            host_key = f"test/mooncake-dispatcher/{os.environ['SGLANG_TEST_DIST_PORT']}"
            store.set(f"{host_key}/{rank}", socket.gethostname())
            hostnames = [
                store.get(f"{host_key}/{peer}").decode() for peer in range(ACTIVE)
            ]
            assert len(set(hostnames)) == ACTIVE, hostnames
            ElasticEPStateManager.init(server_args)
            group = get_moe_ep_group().device_group
            assert dist.get_backend(group) == "mooncake"
            state = ElasticEPStateManager.instance()
            assert state.active_ranks.tolist() == [1, 1, 0, 0]
            assert get_parallel().max_ep_size == CAPACITY
            assert EPBuffer.get_existing_buffer() is None
            print(
                f"MOONCAKE-DISPATCHER-BOOT: host={socket.gethostname()} rank={rank} "
                f"device={torch.cuda.current_device()} pg={dist.get_backend(group)} "
                f"active={ACTIVE} capacity={CAPACITY}",
                flush=True,
            )
            _run_case(rank, group, return_recv_hook=False)
            _run_case(rank, group, return_recv_hook=True)
            _print_native_maps(rank)
        finally:
            elastic_state = get_resources().buffers.pop("mooncake_ep_state", None)
            if elastic_state is not None:
                buffer = elastic_state.buffer
                elastic_state.buffer = None
                del buffer
            ElasticEPStateManager._instance = None
            if dist.is_initialized():
                torch.cuda.synchronize()
                destroy_model_parallel()
                destroy_distributed_environment()


if __name__ == "__main__":
    unittest.main(verbosity=2)
