# Usage: torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py

import os

import tgate
from tgate import LatteConfig, LattePipeline, LatteTGATEConfig
import torch
import time


def run_base():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    tgate.initialize(42)

    config = LatteConfig()
    pipeline = LattePipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    tgate.initialize(42)

    tgate_config = LatteTGATEConfig(
        spatial_broadcast=True,
        spatial_threshold=[0, 20],
        spatial_gap=2,
        temporal_broadcast=True,
        temporal_threshold=[0, 20],
        temporal_gap=2,
        cross_broadcast=True,
        cross_threshold=[20, 50],
        cross_gap=30,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = LatteConfig(enable_tgate=True, tgate_config=tgate_config)
    pipeline = LattePipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]

    save_path = f"./outputs/latte_{prompt.replace(' ', '_')}_spatial_{config.tgate_config.spatial_threshold}_cross_{config.tgate_config.cross_threshold}_tgate.mp4"
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


def prof_tgate():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    tgate.initialize(42)

    tgate_config = LatteTGATEConfig(
        spatial_broadcast=True,
        spatial_threshold=[0, 20],
        spatial_gap=2,
        temporal_broadcast=True,
        temporal_threshold=[0, 20],
        temporal_gap=2,
        cross_broadcast=True,
        cross_threshold=[20, 50],
        cross_gap=30,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = LatteConfig(enable_tgate=True, tgate_config=tgate_config)
    pipeline = LattePipeline(config)
    res = profile_function(pipeline)    

    with open("outputs/profile_delta.txt", "w") as f:
        for line in res:
            f.write(line + "\n")



if __name__ == "__main__":
    # run_base()
    # run_pab()
    prof_tgate()
