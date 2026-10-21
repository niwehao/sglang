"""Manual MoE EP GSM8K test using Mooncake for both A2A and elastic PG.

Default mode starts and tears down the local test service. To reuse an already
running service, set SGLANG_TEST_EXTERNAL_URL. The fault case also requires
SGLANG_TEST_EXTERNAL_OWNER_PID on the same host as the test; it must name the
service launcher whose process tree contains the target scheduler. The test only
sends SIGTERM to that exact owned scheduler process. For an external multi-node
service, inspect the owner process tree and set SGLANG_TEST_FAULT_SCHEDULER to
the target scheduler's exact full command line. Both requests route to DP0, so
target a non-DP0 scheduler; the default title is only for the local one-node fixture.

Run the regular fault case with:

    python -m pytest -q test/manual/ep/test_mooncake_ep.py::TestMooncakeEPElasticEP::test_gsm8k_fault_1

The TBO fault case requires a separate Mooncake service using eager initial
membership (do not pass --elastic-ep-join-mode recover), started with
--enable-two-batch-overlap --disable-cuda-graph --log-level info and
SGLANG_TBO_DEBUG=1 set before launch. It is external-service-only. Provide
SGLANG_TEST_EXTERNAL_URL, SGLANG_TEST_EXTERNAL_OWNER_PID,
SGLANG_TEST_FAULT_SCHEDULER (the exact full command line), and
SGLANG_TEST_EXTERNAL_LOG_PATH (a readable log file containing the survivor DP
worker output for that run). Use the same SGLANG_EP_TEST_MODEL for server and
test client. It observes TBO batch preparation, not a native instruction/CQ
in-flight point. The public HTTP API does not expose the exact active-rank mask.

    python -m pytest -q test/manual/ep/test_mooncake_ep.py::TestMooncakeEPElasticTBO::test_gsm8k_fault_during_tbo_decode
"""

import concurrent.futures
import os
import re
import signal
import threading
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from test_nixl_ep import (
    DP_ATTN,
    ELASTIC_MOONCAKE,
    EP_TEST_MODEL,
    MOONCAKE_COMMON,
    _EPTestBase,
    _find_owned_scheduler,
)

os.environ.setdefault("SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")


class TestMooncakeEPElasticEP(_EPTestBase):
    test_model = EP_TEST_MODEL
    server_args = [
        *MOONCAKE_COMMON,
        *DP_ATTN,
        *ELASTIC_MOONCAKE,
        "--moe-dense-tp-size",
        "1",
        "--enable-dp-lm-head",
    ]
    fault_scheduler_process_title = "sglang::scheduler_DP1_TP1_EP1"

    def test_gsm8k_fault_1(self):
        self._run_fault_and_score()


_TBO_DECODE_LOG = re.compile(
    rf"TboForwardBatchPreparer\.prepare .*?tbo_split_seq_index=(?P<split>\d+)"
    rf".*?\bbs=(?P<bs>\d+)\b.*?forward_mode=(?:ForwardMode\.DECODE|{ForwardMode.DECODE.value})\b"
)


class TestMooncakeEPElasticTBO(unittest.TestCase):
    """External-only TBO fault client; server_args documents the service setup."""

    test_model = EP_TEST_MODEL
    server_args = [
        *TestMooncakeEPElasticEP.server_args,
        "--enable-two-batch-overlap",
        "--disable-cuda-graph",
        "--log-level",
        "info",
    ]

    @classmethod
    def setUpClass(cls):
        cls.model = EP_TEST_MODEL
        cls.base_url = os.environ["SGLANG_TEST_EXTERNAL_URL"].rstrip("/")
        cls.process = None
        cls._service_owner_pid = int(os.environ["SGLANG_TEST_EXTERNAL_OWNER_PID"])
        cls.fault_scheduler_process_title = os.environ["SGLANG_TEST_FAULT_SCHEDULER"]
        cls.external_log_path = Path(os.environ["SGLANG_TEST_EXTERNAL_LOG_PATH"])

    def _submit_pair(self, pool, max_new_tokens):
        barrier = threading.Barrier(2)

        def send(index):
            barrier.wait(timeout=30)
            return requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": f"Write a detailed observatory story, request {index}.",
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                        "ignore_eos": True,
                    },
                    "routed_dp_rank": 0,
                },
                timeout=300,
            )

        return [pool.submit(send, index) for index in range(2)]

    def _wait_log(self, offset, predicate, timeout=120, require_live=None):
        position = offset
        pending = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            size = self.external_log_path.stat().st_size
            if size < position:
                raise RuntimeError("external service log rotated during the test")
            with self.external_log_path.open("rb") as log_file:
                log_file.seek(position)
                data = log_file.read()
                position = log_file.tell()
            lines = (pending + data).split(b"\n")
            pending = lines.pop()
            for raw_line in lines:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
                if predicate(line):
                    print(f"MOONCAKE-TBO-OBSERVED: {line}", flush=True)
                    return line
            if require_live is not None and any(
                future.done() for future in require_live
            ):
                raise AssertionError(
                    "both concurrent requests must remain live until the "
                    "dual-decode log marker appears"
                )
            time.sleep(0.05)
        raise TimeoutError("timed out waiting for the external service log marker")

    def _has_dual_decode_log(self, line):
        if re.search(r"\bDP0\b", line) is None:
            return False
        match = _TBO_DECODE_LOG.search(line)
        if match is None:
            return False
        split, batch_size = int(match.group("split")), int(match.group("bs"))
        return batch_size >= 2 and 0 < split < batch_size

    def test_gsm8k_fault_during_tbo_decode(self):
        warmup_offset = self.external_log_path.stat().st_size
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            warmup = self._submit_pair(pool, max_new_tokens=32)
            self._wait_log(warmup_offset, self._has_dual_decode_log)
            for future in warmup:
                response = future.result(timeout=300)
                self.assertEqual(response.status_code, 200, response.text)

        fault_offset = self.external_log_path.stat().st_size
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            requests_in_flight = self._submit_pair(pool, max_new_tokens=256)
            self._wait_log(
                fault_offset,
                self._has_dual_decode_log,
                require_live=requests_in_flight,
            )
            self.assertTrue(
                all(not future.done() for future in requests_in_flight),
                "the two decode requests must still be live when fault is sent",
            )
            target = _find_owned_scheduler(
                self._service_owner_pid, self.fault_scheduler_process_title
            )
            target.send_signal(signal.SIGTERM)
            self._wait_log(
                fault_offset,
                lambda line: (
                    re.search(r"\bDP0\b", line) is not None
                    and "EPLB due to rank faults" in line
                ),
            )
            for future in requests_in_flight:
                response = future.result(timeout=300)
                self.assertEqual(response.status_code, 200, response.text)

        metrics = _EPTestBase._run_gsm8k(self)
        self.assertGreater(metrics["score"], 0.60)


if __name__ == "__main__":
    unittest.main()
