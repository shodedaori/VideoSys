import torch.distributed as dist
from utils import eval_dist_init
import os
import time
import argparse

import torch
from videosys import CogVideoXConfig, VideoSysEngine
from videosys.pipelines.open_sora import OpenSoraConfig
from videosys.pipelines.cogvideox import CogVideoXConfig
from videosys.pipelines.latte import LatteConfig
from videosys.pipelines.cogvideox.pipeline_tau_cogvideox import CogVideoXTauConfig
from videosys.pipelines.latte.pipeline_stu_latte import LatteSTUConfig
from videosys.utils.utils import set_seed


def get_this_time():
    torch.cuda.synchronize()
    return time.time()

def profile_function(engine):
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


def profile_actor(file_path, config):
    engine = VideoSysEngine(config)
    output = profile_function(engine)
    with open(file_path, "w") as f:
        for line in output:
            f.write(line + "\n")


def get_file_name_and_config_from_string(model, method):
    if model == "opensora":
        if method == "base":
            name = "opensora_base"
            config = OpenSoraConfig(num_sampling_steps=32)
        elif method == "pab":
            name = "opensora_pab"
            config = OpenSoraConfig(num_sampling_steps=32, enable_pab=True)
        elif method == "stu_1/2":
            name = "opensora_stu_1/2"
            config = OpenSoraConfig(num_sampling_steps=32, enable_ti=True, ti_coef=1/2, ti_filter='shortterm')
        elif method == "stu_1/3":
            name = "opensora_stu_1/3"
            config = OpenSoraConfig(num_sampling_steps=32, enable_ti=True, ti_coef=1/3, ti_filter='shortterm')
        elif method == "stu_1/5":
            name = "opensora_stu_1/5"
            config = OpenSoraConfig(num_sampling_steps=32, enable_ti=True, ti_coef=1/5, ti_filter='shortterm')
        else:
            raise ValueError("Invalid method for opensora")
    elif model == "cogvideox":
        if method == "base":
            name = "cogvideox_base"
            config = CogVideoXConfig()
        elif method == "pab":
            name = "cogvideox_pab"
            config = CogVideoXConfig(enable_pab=True)
        elif method == "stu_1/2":
            name = "cogvideox_stu_1/2"
            config = CogVideoXTauConfig(ti_coef=1/2, ti_filter='shortterm')
        elif method == "stu_1/3":
            name = "cogvideox_stu_1/3"
            config = CogVideoXTauConfig(ti_coef=1/3, ti_filter='shortterm')
        elif method == "stu_1/5":
            name = "cogvideox_stu_1/5"
            config = CogVideoXTauConfig(ti_coef=1/5, ti_filter='shortterm')
        else:
            raise ValueError("Invalid method for cogvideox")
    elif model == "latte":
        if method == "base":
            name = "latte_base"
            config = LatteConfig()
        elif method == "pab":
            name = "latte_pab"
            config = LatteConfig(enable_pab=True)
        elif method == "stu_1/2":
            name = "latte_stu_1/2"
            config = LatteSTUConfig(stu_coef=1/2, stu_filter='shortterm')
        elif method == "stu_1/3":
            name = "latte_stu_1/3"
            config = LatteSTUConfig(stu_coef=1/3, stu_filter='shortterm')
        elif method == "stu_1/5":
            name = "latte_stu_1/5"
            config = LatteSTUConfig(stu_coef=1/5, stu_filter='shortterm')
        else:
            raise ValueError("Invalid method for latte")
    else:
        raise ValueError("Invalid model name")
    
    print(f"Model: {model}, Method: {method}, Name: {name}")

    return (name, config)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="opensora", help="Model name: opensora, cogvideox, latte")
    argparser.add_argument("--method", type=str, default="base", help="Method name: base, pab, stu_1/2, stu_1/3, stu_1/5")
    args = argparser.parse_args()
    name, config = get_file_name_and_config_from_string(args.model, args.method)

    file_path = os.path.join("profile", name)

    eval_dist_init()
    profile_actor(file_path, config)
    dist.destroy_process_group()
