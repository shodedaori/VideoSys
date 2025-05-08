import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import time
import torch
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

    delta_config = CogvideoxDELTAConfig(
        steps=10,
        delta_skip=True,
        # delta_threshold={(0, 2): [0, 2]},
        delta_threshold={(0, 5): [0, 5]},
        delta_gap=2,
    )

    config = CogVideoXConfig(enable_delta=True, delta_config=delta_config)
    pipeline = CogVideoXPipeline(config)
    res = profile_function(pipeline)    

    with open("outputs/profile_delta.txt", "w") as f:
        for line in res:
            f.write(line + "\n")



if __name__ == "__main__":
    # run_base()
    # run_delta()
    prof_delta()
