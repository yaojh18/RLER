"""Register qwen3_5 / qwen3_5_text model_types with transformers AutoConfig.

The Qwen3.5 family (Qwen3.5-9B, Qwen3.5-27B, ...) is missing from the
baked container's transformers 4.57.x. mbridge.AutoBridge.from_pretrained
calls AutoConfig.from_pretrained internally and dies with
    ValueError: ... model type `qwen3_5` ... not recognize ...
before Qwen3_5Bridge ever runs.

Subclassing Qwen3Config keeps AutoConfig happy -- it only routes by
model_type. The actual arch (hidden/layers/ffn) comes from MODEL_ARGS
(megatron CLI), not from this object, so the convert path doesn't care
that Qwen3Config skips qwen3.5-only fields like attn_output_gate or
layer_types. Imported eagerly from slime_plugins.mbridge.__init__ so any
caller of `import slime_plugins.mbridge` gets the registration.

Note on __init__: transformers' from_dict log call serializes the new
config via to_diff_dict, which iterates sub-attrs and calls
`obj.__class__().to_dict()` on each. If text_config arrives as a raw
dict from config.json and we leave it as a dict, that call becomes
`dict().to_dict()` -> AttributeError. We explicitly wrap it into
Qwen3_5TextConfig so to_diff_dict has a real config instance to walk.
"""

from transformers import AutoConfig
from transformers.models.qwen3 import Qwen3Config


class Qwen3_5TextConfig(Qwen3Config):
    model_type = "qwen3_5_text"


class Qwen3_5Config(Qwen3Config):
    model_type = "qwen3_5"
    sub_configs = {"text_config": Qwen3_5TextConfig}

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Qwen3_5TextConfig(**text_config)
        super().__init__(**kwargs)
        self.text_config = text_config


AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
