"""Server entry point: hide the GPUs the profile doesn't use, THEN import the server.

The CUDA driver reads CUDA_VISIBLE_DEVICES once, at its first initialisation, and importing
transformers/peft already initialises it (torch.cuda.is_available()). So the restriction must
happen before `zroform.api` is imported; this module imports only the config until then.
  python -m zroform.launch --host 127.0.0.1 --port 8100     (profile from $ZROFORM_PROFILE)
"""

from __future__ import annotations

import argparse
import dataclasses
import os

from zroform.config import load_profile, restrict_visible_gpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    profile = load_profile(os.environ.get("ZROFORM_PROFILE", "profiles/node8x-qwen38-27b-ctx32k.yaml"))
    if profile.trainer_cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = profile.trainer_cuda_alloc_conf
    if profile.trainer_device.startswith("cuda"):
        device, gpus = restrict_visible_gpus(profile.trainer_device, profile.sampler_gpus)
        profile = dataclasses.replace(profile, trainer_device=device, sampler_gpus=gpus)

    import uvicorn

    from zroform.api import create_app
    uvicorn.run(create_app(profile), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
