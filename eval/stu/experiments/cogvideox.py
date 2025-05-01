import torch.distributed as dist
from utils import dist_generate_func, generate_func, read_prompt_list, eval_dist_init

from videosys import CogVideoXConfig, VideoSysEngine
from videosys.pipelines.cogvideox.pipeline_tau_cogvideox import CogVideoXTauConfig

NUM_SAMPLES = 4
PROMPT_PATH = "vbench/VBench_full_info.json"


def eval_base(prompt_list):
    config = CogVideoXConfig()
    engine = VideoSysEngine(config)
    dist_generate_func(engine, prompt_list, "./samples/cogvideox_base", loop=NUM_SAMPLES)


def eval_pab(prompt_list):
    config = CogVideoXConfig(enable_pab=True)
    engine = VideoSysEngine(config)
    dist_generate_func(engine, prompt_list, "./samples/cogvideox_pab", loop=NUM_SAMPLES)


def eval_stu(prompt_list):
    config = CogVideoXTauConfig(
        ti_coef=0.5,
        ti_filter='pframe'
    )
    engine = VideoSysEngine(config)
    dist_generate_func(engine, prompt_list, "./samples/cogvideox_stu", loop=NUM_SAMPLES)


if __name__ == "__main__":

    eval_dist_init()

    prompt_list = read_prompt_list(PROMPT_PATH)
    # eval_base(prompt_list)
    # eval_pab(prompt_list)
    eval_stu(prompt_list)

    # dist.destroy_process_group()
