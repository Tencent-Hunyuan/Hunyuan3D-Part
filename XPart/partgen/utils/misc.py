import os
import torch
import logging
import importlib
from typing import Union
from functools import wraps

from omegaconf import OmegaConf, DictConfig, ListConfig


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger("hy3dgen.partgen")


class synchronize_timer:
    """Synchronized timer to count the inference time of `nn.Module.forward`.

    Supports both context manager and decorator usage.

    Example as context manager:
    ```python
    with synchronize_timer('name') as t:
        run()
    ```

    Example as decorator:
    ```python
    @synchronize_timer('Export to trimesh')
    def export_to_trimesh(mesh_output):
        pass
    ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get("HY3DGEN_DEBUG", "0") == "1":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get("HY3DGEN_DEBUG", "0") == "1":
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f"{self.name} takes {self.time} ms")

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def get_config_from_file(config_file: str) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)

    if "base_config" in config_file.keys():
        if config_file["base_config"] == "default_base":
            base_config = OmegaConf.create()
            # base_config = get_default_config()
        elif config_file["base_config"].endswith(".yaml"):
            base_config = get_config_from_file(config_file["base_config"])
        else:
            raise ValueError(
                f"{config_file} must be `.yaml` file or it contains `base_config` key."
            )

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    if config.get("from_pretrained", None):
        return cls.from_pretrained(
            config["from_pretrained"],
            use_safetensors=config.get("use_safetensors", False),
            variant=config.get("variant", "fp16"),
        )

    params = config.get("params", dict())
    # params.update(kwargs)
    # instance = cls(**params)
    kwargs.update(params)
    instance = cls(**kwargs)

    return instance


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def instantiate_non_trainable_model(config):
    model = instantiate_from_config(config)
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False

    return model


def smart_load_model(
    model_path,
):
    original_model_path = model_path
    # try local path
    base_dir = os.environ.get("HY3DGEN_MODELS", "~/.cache/xpart")
    model_fld = os.path.expanduser(os.path.join(base_dir, model_path))
    logger.info(f"Try to load model from local path: {model_path}")
    if not os.path.exists(model_path):
        logger.info("Model path not exists, try to download from huggingface")
        try:
            from huggingface_hub import snapshot_download

            # 只下载指定子目录
            path = snapshot_download(
                repo_id=original_model_path,
                # allow_patterns=[f"{subfolder}/*"],  # 关键修改：模式匹配子文件夹
                local_dir=model_fld,
            )
            model_path = path  # os.path.join(path, subfolder)  # 保持路径拼接逻辑不变
        except ImportError:
            logger.warning(
                "You need to install HuggingFace Hub to load models from the hub."
            )
            raise RuntimeError(f"Model path {model_path} not found")
        except Exception as e:
            raise e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {original_model_path} not found")

    return model_path


def init_from_ckpt(model, ckpt, prefix="model", ignore_keys=()):
    if "state_dict" not in ckpt:
        # deepspeed ckpt
        state_dict = {}
        ckpt = ckpt["module"] if "module" in ckpt else ckpt
        for k in ckpt.keys():
            new_k = k.replace("_forward_module.", "")
            state_dict[new_k] = ckpt[k]
    else:
        state_dict = ckpt["state_dict"]
    keys = list(state_dict.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik in k:
                print("Deleting key {} from state_dict.".format(k))
                del state_dict[k]
    state_dict = {
        k.replace(prefix + ".", ""): v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
        print(f"Unexpected Keys: {unexpected}")
