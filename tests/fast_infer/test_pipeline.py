import pytest

import torch

from videosys import OpenSoraConfig, VideoSysEngine
from videosys.models.transformers.cache_model import STDiT3C
from videosys.schedulers.scheduling_si_open_sora import SIRFLOW
from videosys.utils.test import empty_cache


@empty_cache
def test_si(num_gpus):
    config = OpenSoraConfig(num_gpus=num_gpus, use_si=True, si_sparsity=0.75)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_{num_gpus}.mp4")


@empty_cache
def test_base(num_gpus):
    config = OpenSoraConfig(num_gpus=num_gpus)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_{num_gpus}.mp4")


if __name__ == '__main__':
    # test_base(1)
    test_si(1)
