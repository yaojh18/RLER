import logging
from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


logger = logging.getLogger(__name__)


def get_auto_bridge():
    try:
        from megatron.bridge import AutoBridge
    except ModuleNotFoundError as exc:
        if exc.name != "megatron.bridge":
            raise
        from mbridge import AutoBridge

        logger.warning("megatron.bridge is unavailable; falling back to mbridge.AutoBridge")
        return AutoBridge

    try:
        import slime_plugins.megatron_bridge  # noqa: F401
    except ModuleNotFoundError as exc:
        logger.warning("Skipping slime custom megatron bridge registration: %s", exc)

    return AutoBridge


def load_auto_bridge_from_hf(path, **kwargs):
    auto_bridge = get_auto_bridge()
    if hasattr(auto_bridge, "from_hf_pretrained"):
        return auto_bridge.from_hf_pretrained(path, **kwargs)
    return auto_bridge.from_pretrained(path, **kwargs)


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
