from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import hf_hub_download
import json


def _parse_shell_model_args(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    flags: dict[str, str] = {}
    for key in [
        "--num-layers",
        "--hidden-size",
        "--ffn-hidden-size",
        "--num-attention-heads",
        "--num-query-groups",
        "--kv-channels",
        "--vocab-size",
        "--rotary-base",
        "--norm-epsilon",
    ]:
        match = re.search(rf"{re.escape(key)}\s+([^\s]+)", text)
        assert match is not None, f"Missing {key} in {path}"
        flags[key] = match.group(1)
    return flags


def test_qwen35_9b_model_args_match_hf_config():
    config_path = Path(__file__).resolve().parents[2] / "swe_agent" / "configs" / "qwen3.5-9B.sh"
    flags = _parse_shell_model_args(config_path)

    hf_config_path = hf_hub_download(repo_id="Qwen/Qwen3.5-9B", filename="config.json", local_files_only=False)
    hf_config = json.loads(Path(hf_config_path).read_text(encoding="utf-8"))
    text_config = hf_config["text_config"]

    assert flags["--num-layers"] == str(text_config["num_hidden_layers"])
    assert flags["--hidden-size"] == str(text_config["hidden_size"])
    assert flags["--ffn-hidden-size"] == str(text_config["intermediate_size"])
    assert flags["--num-attention-heads"] == str(text_config["num_attention_heads"])
    assert flags["--num-query-groups"] == str(text_config["num_key_value_heads"])
    assert flags["--kv-channels"] == str(text_config["head_dim"])
    assert flags["--vocab-size"] == str(text_config["vocab_size"])
    assert float(flags["--norm-epsilon"]) == float(text_config["rms_norm_eps"])
    assert float(flags["--rotary-base"]) == float(text_config["rope_parameters"]["rope_theta"])
