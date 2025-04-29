import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import deltadit
from deltadit import OpenSoraConfig, OpenSoraDELTAConfig, OpenSoraPipeline

from deltadit import CogVideoXConfig, CogVideoXPipeline, CogvideoxDELTAConfig


def run_base():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    deltadit.initialize(42)

    config = CogVideoXConfig()
    pipeline = CogVideoXPipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt, seed=0).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_delta():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    deltadit.initialize(42)

    delta_config = CogvideoxDELTAConfig(
        steps=10,
        delta_skip=True,
        # delta_threshold={(0, 2): [0, 2]},
        delta_threshold={(0, 5): [0, 5]},
        delta_gap=2,
    )
    config = CogVideoXConfig(enable_delta=True, delta_config=delta_config)
    pipeline = CogVideoXPipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt, seed=0).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}_delta.mp4")


if __name__ == "__main__":
    # run_base()
    run_delta()
