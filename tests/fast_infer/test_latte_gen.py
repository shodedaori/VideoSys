import pytest

from videosys import LatteConfig, VideoSysEngine
from videosys.pipelines.latte.pipeline_stu_latte import LatteSTUConfig
from videosys.utils.test import empty_cache


@empty_cache
def test_base(prompt, num_infer_steps=50, measure_flag=False):
    config = LatteConfig()
    engine = VideoSysEngine(config)

    if measure_flag:
        # for warming up
        video = engine.generate(prompt, num_inference_steps=num_infer_steps, seed=0).video[0]
    
    video = engine.generate(prompt, num_inference_steps=num_infer_steps, seed=0).video[0]
    
    if not measure_flag:
        output_title = prompt.replace(" ", "_")[:20]
        engine.save_video(video, f"./test_outputs/Latte/{output_title}_base.mp4")


@empty_cache
def test_stu(prompt, num_infer_steps=50, stu_rate=0.5, measure_flag=False):
    stu_filter = 'shortterm'
    config = LatteSTUConfig(
        stu_coef=stu_rate,
        stu_filter=stu_filter,
    )
    engine = VideoSysEngine(config)

    if measure_flag:
        # for warming up
        video = engine.generate(prompt, num_inference_steps=num_infer_steps, seed=0).video[0]

    video = engine.generate(prompt, num_inference_steps=num_infer_steps, seed=0).video[0]

    if not measure_flag:
        output_title = prompt.replace(" ", "_")[:20]
        engine.save_video(video, f"./test_outputs/Latte/{output_title}_{stu_filter}_{stu_rate}.mp4")


if __name__ == "__main__":
    prompt = "A snowy forest landscape with a dirt road running through it. The road is flanked by trees covered in snow, and the ground is also covered in snow. The sun is shining, creating a bright and serene atmosphere. The road appears to be empty, and there are no people or animals visible in the video. The style of the video is a natural landscape shot, with a focus on the beauty of the snowy forest and the peacefulness of the road."

    # test_base(prompt)
    # test_stu(prompt)

    steps = [25, 50, 100]
    coefs = [1/2, 1/3, 1/5]

    for step in steps:
        print(f"Testing Latte base with {step} steps")
        test_base(prompt, num_infer_steps=step, measure_flag=True)

    for step in steps:
        for coef in coefs:
            print(f"Testing Latte STU with {step} steps and coef {coef}")
            test_stu(prompt, num_infer_steps=step, stu_rate=coef, measure_flag=True)
