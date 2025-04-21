import pytest

from videosys import CogVideoXConfig, VideoSysEngine
from videosys.pipelines.cogvideox import CogVideoXTauConfig, CogVideoXConfig
from videosys.utils.test import empty_cache


@pytest.mark.parametrize("num_gpus", [1, 2])
@pytest.mark.parametrize("model_path", ["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"])
@empty_cache
def test_base(num_gpus, model_path):
    config = CogVideoXConfig(
        model_path=model_path, 
        num_gpus=num_gpus
    )
    engine = VideoSysEngine(config)

    # prompt = "Sunset over the sea."
    # prompt = "A person is petting an animal (not a cat)"
    prompt = "A person is petting a dog"
    # prompt = "A panda standing on a surfboard in the ocean in sunset."
    # prompt = "A sleek, modern laptop, its screen displaying a vibrant, paused scene, sits on a minimalist wooden desk. The room is bathed in soft, natural light filtering through sheer curtains, casting gentle shadows. The laptop's keyboard is mid-illumination, with a faint glow emanating from the keys, suggesting a moment frozen in time. Dust particles are suspended in the air, caught in the light, adding to the stillness. A steaming cup of coffee beside the laptop remains untouched, with wisps of steam frozen in mid-air. The scene captures a serene, almost magical pause in an otherwise bustling workspace."

    video = engine.generate(prompt, num_inference_steps=50, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/cogvideo_base_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1, 2])
@pytest.mark.parametrize("model_path", ["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"])
@empty_cache
def test_tau(num_gpus, model_path):
    my_filter = 'pframe'
    config = CogVideoXTauConfig(
        model_path=model_path, 
        num_gpus=num_gpus,
        ti_coef=0.4,
        ti_filter=my_filter
    )
    engine = VideoSysEngine(config)

    # prompt = "Sunset over the sea."
    # prompt = "A person is petting an animal (not a cat)"
    prompt = "A person is petting a dog"
    # prompt = "A panda standing on a surfboard in the ocean in sunset."
    # prompt = "A sleek, modern laptop, its screen displaying a vibrant, paused scene, sits on a minimalist wooden desk. The room is bathed in soft, natural light filtering through sheer curtains, casting gentle shadows. The laptop's keyboard is mid-illumination, with a faint glow emanating from the keys, suggesting a moment frozen in time. Dust particles are suspended in the air, caught in the light, adding to the stillness. A steaming cup of coffee beside the laptop remains untouched, with wisps of steam frozen in mid-air. The scene captures a serene, almost magical pause in an otherwise bustling workspace."

    video = engine.generate(prompt, num_inference_steps=50, seed=0, verbose=True).video[0]
    engine.save_video(video, f"./test_outputs/{model_path}_tau_{my_filter}.mp4")


if __name__ == '__main__':
    # test_base(1, "THUDM/CogVideoX-2b")
    test_tau(1, "THUDM/CogVideoX-2b")
