import torch.distributed as dist
from utils import dist_generate_func, generate_func, read_prompt_list, eval_dist_init

from videosys import OpenSoraConfig, OpenSoraPABConfig, VideoSysEngine

NUM_SAMPLES = 2
PROMPT_PATH = "vbench/small_vbench.json"


def eval_base(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    dist_generate_func(engine, prompt_list, "./samples/opensora_base", loop=NUM_SAMPLES)


def eval_pab1(prompt_list):
    config = OpenSoraConfig(enable_pab=True)
    engine = VideoSysEngine(config)
    dist_generate_func(engine, prompt_list, "./samples/opensora_pab", loop=NUM_SAMPLES)


if __name__ == "__main__":

    eval_dist_init()

    prompt_list = read_prompt_list(PROMPT_PATH)
    # eval_base(prompt_list)
    eval_pab1(prompt_list)

    dist.destroy_process_group()
