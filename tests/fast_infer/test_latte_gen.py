import pytest

from videosys import LatteConfig, VideoSysEngine
from videosys.utils.test import empty_cache


@empty_cache
def test_base():
    config = LatteConfig()
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    output_title = prompt.replace(" ", "_")[:20]
    engine.save_video(video, f"./test_outputs/Latte/{output_title}_base.mp4")


if __name__ == "__main__":
    test_base()
