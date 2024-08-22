from utils import generate_func, read_prompt_list

import opendit
from opendit import OpenSoraConfig, OpenSoraPipeline
from opendit.models.opensora import OpenSoraPABConfig


def wo_spatial(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_broadcast=False)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/components_ablation/wo_spatial")


def wo_temporal(prompt_list):
    pab_config = OpenSoraPABConfig(temporal_broadcast=False)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/components_ablation/wo_temporal")


def wo_cross(prompt_list):
    pab_config = OpenSoraPABConfig(cross_broadcast=False)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/components_ablation/wo_cross")


def wo_mlp(prompt_list):
    pab_config = OpenSoraPABConfig(mlp_skip=False)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/components_ablation/wo_mlp")


if __name__ == "__main__":
    opendit.initialize(42)
    prompt_list = read_prompt_list("./vbench/VBench_full_info.json")
    wo_spatial(prompt_list)
    wo_temporal(prompt_list)
    wo_cross(prompt_list)
    wo_mlp(prompt_list)