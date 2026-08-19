# RLER agent

This package contains the SWE-agent rollout implementation and the reusable
`agent_rl` training/runtime interfaces.

Install it from this directory:

```bash
uv pip install -e .
```

The main rollout entry points are:

- `swe_agent/run/run_swe_agent.py`: ordinary SWE-agent rollout and evaluation.
- `swe_agent/run/search_swe_agent.py`: trajectory search.
- `swe_agent/run/aggregate_swe_agent.py`: aggregation over existing trajectories.
- `agent_rl/`: model-service, rollout protocol, and local vLLM runtime support
  shared by training and rollout entry points.

The retired deep-research MCP agent, its data-build utilities, and its benchmark
evaluation package are intentionally not part of this package.
