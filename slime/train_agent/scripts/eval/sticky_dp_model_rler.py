"""Sticky-DP wrapper around the RLER fork's LitellmTextbasedModel.

Mirror of `sticky_dp_model.py` (which wraps upstream `minisweagent`) but
subclasses RLER's `swe_agent.models.litellm_textbased_model.LitellmTextbasedModel`
so it composes with the 50-sample slurm (`eval_pass_at_4_9b_v2.slurm`) that
runs the RLER fork end-to-end for the testbed-fix prompts in
`swebench_backticks.yaml`.

Pins each per-instance model object to a single sglang DP rank via
`data_parallel_rank` in `extra_body`, so the multi-turn prompt prefix stays
in one rank's radix KV cache. mini-swe-agent calls `get_model(...)` once per
benchmark instance inside `process_instance`, so a global round-robin
counter at __init__ time gives each instance a stable rank for the whole
trajectory. The 4 attempts of one instance (synthetic ids `__a1..__a4`) end
up on 4 distinct ranks — desirable, since their per-attempt prefixes diverge
after step 1 and would otherwise contend on a single rank's radix cache.

Env vars:
  STICKY_DP_SIZE  number of DP ranks (default 8). 0 disables routing.
"""

import itertools
import os
import threading

from swe_agent.models.litellm_textbased_model import LitellmTextbasedModel

_DP_COUNTER = itertools.count()
_LOCK = threading.Lock()


def _next_dp_rank(dp_size: int) -> int:
    with _LOCK:
        return next(_DP_COUNTER) % dp_size


class StickyDPLitellmTextbasedModel(LitellmTextbasedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dp_size = int(os.environ.get("STICKY_DP_SIZE", "8"))
        self._dp_rank = _next_dp_rank(dp_size) if dp_size > 0 else None

    def _query(self, messages, **kwargs):
        if self._dp_rank is not None:
            eb = dict(kwargs.get("extra_body") or {})
            eb.setdefault("data_parallel_rank", self._dp_rank)
            kwargs["extra_body"] = eb
        return super()._query(messages, **kwargs)
