import os
import signal
import time
import unittest
from types import SimpleNamespace

import psutil
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

EP_TEST_MODEL = os.environ.get("SGLANG_EP_TEST_MODEL", DEFAULT_MODEL_NAME_FOR_TEST_MLA)
TEST_MODEL = os.environ.get("NIXL_EP_TEST_MODEL", EP_TEST_MODEL)
os.environ.setdefault("SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")

ib_devices = get_rdma_devices_args()

NIXL_COMMON = [
    "--trust-remote-code",
    "--moe-a2a-backend",
    "nixl",
    "--deepep-mode",
    "low_latency",
    "--tp",
    "8",
    "--mem-fraction-static",
    "0.78",
]
DP_ATTN = ["--attn-dp-size", "8"]
ELASTIC_NIXL = [
    "--elastic-ep-backend",
    "nixl",
    "--enable-eplb",
    "--ep-num-redundant-experts",
    "24",
]
ELASTIC_MOONCAKE = [
    "--elastic-ep-backend",
    "mooncake",
    "--mooncake-ib-device",
    ib_devices,
    "--enable-eplb",
    "--ep-num-redundant-experts",
    "24",
]
MOONCAKE_COMMON = [
    "--trust-remote-code",
    "--moe-a2a-backend",
    "mooncake",
    "--deepep-mode",
    "low_latency",
    "--tp",
    "8",
    "--mem-fraction-static",
    "0.78",
]


def _find_owned_scheduler(owner_pid: int, process_title: str) -> psutil.Process:
    owner = psutil.Process(owner_pid)
    candidates = []
    for process in [owner, *owner.children(recursive=True)]:
        try:
            if " ".join(process.cmdline()).rstrip() == process_title:
                candidates.append(process)
        except (psutil.Error, OSError):
            continue
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one owned scheduler {process_title!r} below pid {owner_pid}, "
            f"found {len(candidates)}. Refusing to signal an unverified process."
        )
    return candidates[0]


class _EPTestBase(CustomTestCase):
    server_args: list[str] = []
    test_model = TEST_MODEL

    @classmethod
    def setUpClass(cls):
        cls.model = cls.test_model
        external_url = os.environ.get("SGLANG_TEST_EXTERNAL_URL")
        if external_url:
            cls.base_url = external_url.rstrip("/")
            cls.process = None
            owner_pid = os.environ.get("SGLANG_TEST_EXTERNAL_OWNER_PID")
            cls._service_owner_pid = int(owner_pid) if owner_pid else None
            return

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.server_args,
        )
        cls._service_owner_pid = cls.process.pid

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
            cls.process.wait(timeout=15)
            time.sleep(2)

    def _run_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_sgl_eval(args)
        print(metrics)
        return metrics

    def test_gsm8k(self):
        metrics = self._run_gsm8k()
        self.assertGreater(metrics["score"], 0.60)

    def _run_fault_and_score(self):
        owner_pid = self._service_owner_pid
        if owner_pid is None:
            raise RuntimeError(
                "Fault testing an external service requires "
                "SGLANG_TEST_EXTERNAL_OWNER_PID on the target host."
            )
        process_title = os.environ.get(
            "SGLANG_TEST_FAULT_SCHEDULER", self.fault_scheduler_process_title
        )
        process = _find_owned_scheduler(owner_pid, process_title)
        process.send_signal(signal.SIGTERM)
        # Bootstrap one forward on a survivor so the controller learns the
        # post-fault active-rank mask before dispatching concurrent requests.
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 1},
                "routed_dp_rank": 0,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        metrics = self._run_gsm8k()
        self.assertGreater(metrics["score"], 0.60)


class TestNixlEPTP(_EPTestBase):
    server_args = [*NIXL_COMMON]


class TestNixlEPDPAttn(_EPTestBase):
    server_args = [*NIXL_COMMON, *DP_ATTN]


class TestNixlEPElasticEP(_EPTestBase):
    server_args = [*NIXL_COMMON, *DP_ATTN, *ELASTIC_NIXL]


class TestNixlMoeMooncakeElasticEP(_EPTestBase):
    server_args = [
        *NIXL_COMMON,
        *DP_ATTN,
        *ELASTIC_MOONCAKE,
        "--moe-dense-tp-size",
        "1",
        "--enable-dp-lm-head",
    ]
    fault_scheduler_process_title = "sglang::scheduler_DP1_TP1_EP1"

    def test_gsm8k_fault_1(self):
        self._run_fault_and_score()


if __name__ == "__main__":
    unittest.main()
