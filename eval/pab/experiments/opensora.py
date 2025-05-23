from utils import generate_func, read_prompt_list

from videosys import OpenSoraConfig, OpenSoraPABConfig, VideoSysEngine

NUM_SAMPLES = 2
PROMPT_PATH = "vbench/small_vbench.json"


def eval_base(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_base", loop=NUM_SAMPLES)


def eval_pab1(prompt_list):
    config = OpenSoraConfig(enable_pab=True)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_pab1", loop=NUM_SAMPLES)


def eval_pab2(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_gap=3, temporal_gap=5, cross_gap=7)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_pab2", loop=NUM_SAMPLES)


def eval_pab3(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_gap=5, temporal_gap=7, cross_gap=9)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_pab3", loop=NUM_SAMPLES)


if __name__ == "__main__":
    prompt_list = read_prompt_list(PROMPT_PATH)
    eval_base(prompt_list)
    eval_pab1(prompt_list)
    # eval_pab2(prompt_list)
    # eval_pab3(prompt_list)
