import pytest

from videosys import LatteConfig, VideoSysEngine
from videosys.pipelines.latte.pipeline_stu_latte import LatteSTUConfig
from videosys.utils.test import empty_cache


@empty_cache
def test_base():
    config = LatteConfig()
    engine = VideoSysEngine(config)

    # prompt = "Sunset over the sea."
    prompt = "A bear and a zebra are playing chess in a park."
    
    video = engine.generate(prompt, seed=0).video[0]
    output_title = prompt.replace(" ", "_")[:20]
    engine.save_video(video, f"./test_outputs/Latte/{output_title}_base.mp4")


@empty_cache
def test_stu():
    my_filter = 'shortterm'
    config = LatteSTUConfig(
        stu_coef=0.5,
        stu_filter=my_filter,
    )
    engine = VideoSysEngine(config)

    # prompt = "Sunset over the sea."
    prompt = "two bears are playing chess in a park."
    # prompt = "a dog is sitting on a table in a cafe"
    video = engine.generate(prompt, seed=0, verbose=False).video[0]
    output_title = prompt.replace(" ", "_")[:20]
    engine.save_video(video, f"./test_outputs/Latte/{output_title}_{my_filter}.mp4")


if __name__ == "__main__":
    # test_base()
    test_stu()
