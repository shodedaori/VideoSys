import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import tgate
from tgate import OpenSoraConfig, OpenSoraPipeline, OpenSoraTGATEConfig
import torch
import time


def run_base():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    tgate.initialize(42)

    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12358"

    tgate.initialize(42)

    tgate_config = OpenSoraTGATEConfig(
        spatial_broadcast=True,
        spatial_threshold=[0, 12],
        spatial_gap=2,
        temporal_broadcast=True,
        temporal_threshold=[0, 12],
        temporal_gap=2,
        cross_broadcast=True,
        cross_threshold=[12, 30],
        cross_gap=18,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = OpenSoraConfig(enable_tgate=True, tgate_config=tgate_config)
    pipeline = OpenSoraPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]

    save_path = f"./outputs/opensora_{prompt.replace(' ', '_')}_spatial_{config.tgate_config.spatial_threshold}_cross_{config.tgate_config.cross_threshold}_tgate.mp4"
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

    tgate_config = OpenSoraTGATEConfig(
        spatial_broadcast=True,
        spatial_threshold=[0, 12],
        spatial_gap=2,
        temporal_broadcast=True,
        temporal_threshold=[0, 12],
        temporal_gap=2,
        cross_broadcast=True,
        cross_threshold=[12, 30],
        cross_gap=18,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = OpenSoraConfig(enable_tgate=True, tgate_config=tgate_config)
    pipeline = OpenSoraPipeline(config)
    res = profile_function(pipeline)    

    with open("outputs/profile_delta.txt", "w") as f:
        for line in res:
            f.write(line + "\n")


if __name__ == "__main__":
    # torch.backends.cudnn.enabled = False
    # run_base() # 01:17
    # run_pab()  # enable_tgate=False 01:17    # enable_tgate=True 01:07
    prof_tgate() # 01:07
