import json
import os
import tqdm
import torch
import torch.distributed as dist

from videosys.utils.utils import set_seed
from videosys.core.mp_utils import get_distributed_init_method, get_open_port
from videosys.utils.logging import init_dist_logger, logger


def generate_func(pipeline, prompt_list, output_dir, loop: int = 5, kwargs: dict = {}):
    kwargs["verbose"] = False
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            set_seed(l)
            video = pipeline.generate(prompt, **kwargs).video[0]
            pipeline.save_video(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"))


def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list


def eval_dist_init():
    if not dist.is_initialized():
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())
    
    init_dist_logger()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Evaluation world size: {dist.get_world_size()}")


def dist_generate_func(
    pipeline,
    prompt_list,
    output_dir,
    loop: int = 5,
    kwargs: dict = {},
):

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    global_pg = dist.group.WORLD

    local_prompt_list = prompt_list[rank::world_size]
    master_flag = rank == 0

    pipeline.driver_worker._set_parallel(dp_size=world_size, sp_size=1)

    for prompt in tqdm.tqdm(local_prompt_list, disable=not master_flag):
        for l in range(loop):
            set_seed(l)
            video = pipeline.generate(prompt, **kwargs).video[0]
            pipeline.save_video(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"))

    dist.barrier(group=global_pg)
