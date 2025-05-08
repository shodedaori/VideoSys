# Usage: torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import deltadit
from deltadit import LatteConfig, LatteDELTAConfig, LattePipeline
import torch
import time


def run_base():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    deltadit.initialize(42)

    config = LatteConfig()
    pipeline = LattePipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12358"

    deltadit.initialize(42)

    delta_config = LatteDELTAConfig(
        steps=10,
        delta_skip=True,
        # delta_threshold={(0, 2): [0, 2]},
        delta_threshold={(0, 1): [0, 1]},
        delta_gap=2,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = LatteConfig(enable_delta=True, delta_config=delta_config)
    pipeline = LattePipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]
    video = pipeline.generate(prompt).video[0]

    save_path = f"./outputs/latte_delta_{config.delta_config.delta_skip}_{prompt.replace(' ', '_')}_delta_threshold_{config.delta_config.delta_threshold}_delta_gap_{config.delta_config.delta_gap}.mp4"
    pipeline.save_video(video, save_path)
    print(f"Saved video to {save_path}")


def profile_function(engine):
    def get_this_time():
        torch.cuda.synchronize()
        return time.time()

    output_list = []

    prompt = "A snowy forest landscape with a dirt road running through it. The road is flanked by trees covered in snow, and the ground is also covered in snow. The sun is shining, creating a bright and serene atmosphere. The road appears to be empty, and there are no people or animals visible in the video. The style of the video is a natural landscape shot, with a focus on the beauty of the snowy forest and the peacefulness of the road."

    video = engine.generate(prompt, seed=0).video[0]

    with torch.profiler.profile(with_flops=True) as p:
        video = engine.generate(prompt).video[0]
    output_list.append('{:.2f} TFLOPS (torch profile)'.format(sum(k.flops for k in p.key_averages()) / 1e12))

    start = get_this_time()
    video = engine.generate(prompt).video[0]
    end = get_this_time()
    output_list.append(f"Time taken for OpenSora base: {end - start:.2f} seconds")
    
    return output_list


def prof_delta():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    deltadit.initialize(42)

    delta_config = LatteDELTAConfig(
        steps=10,
        delta_skip=True,
        # delta_threshold={(0, 2): [0, 2]},
        delta_threshold={(0, 1): [0, 1]},
        delta_gap=2,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = LatteConfig(enable_delta=True, delta_config=delta_config)
    pipeline = LattePipeline(config)
    res = profile_function(pipeline)    

    with open("outputs/profile_delta.txt", "w") as f:
        for line in res:
            f.write(line + "\n")


if __name__ == "__main__":
    # run_base()
    # run_pab()
    prof_delta()
