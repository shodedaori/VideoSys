import argparse
from utils import generate_func, read_prompt_list

from videosys import OpenSoraConfig, VideoSysEngine

NUM_SAMPLES = 4
PROMPT_PATH = "vbench/middle_vbench.json"

config_list = [
    dict(),
    # constant examples
    dict(enable_ti=True, ti_coef=0.4, ti_filter='constant'),
    dict(enable_ti=True, ti_coef=0.6, ti_filter='constant'),
    dict(enable_ti=True, ti_coef=0.8, ti_filter='constant'),
    # weighted threshold examples
    dict(enable_ti=True, ti_coef=0.8, ti_filter='weighted_threshold'),
    dict(enable_ti=True, ti_coef=1.0, ti_filter='weighted_threshold'),
    dict(enable_ti=True, ti_coef=1.2, ti_filter='weighted_threshold'),
    dict(enable_ti=True, ti_coef=1.4, ti_filter='weighted_threshold'),
]

file_path = [
    "./samples/opensora_base",
    "./samples/opensora_ti_constant_0.4",
    "./samples/opensora_ti_constant_0.6",
    "./samples/opensora_ti_constant_0.8",
    "./samples/opensora_ti_weighted_threshold_0.8",
    "./samples/opensora_ti_weighted_threshold_1.0",
    "./samples/opensora_ti_weighted_threshold_1.2",
    "./samples/opensora_ti_weighted_threshold_1.4",
]


def eval_base(prompt_list, file_path, **kwargs):
    config = OpenSoraConfig(**kwargs)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, file_path, loop=NUM_SAMPLES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluation")
    parser.add_argument("--index", default=0, help="The index of the configuration.", type=int)
    args = vars(parser.parse_args())

    file_path = file_path[args['index']]
    config = config_list[args['index']]

    prompt_list = read_prompt_list(PROMPT_PATH)
    eval_base(prompt_list, file_path, **config)
