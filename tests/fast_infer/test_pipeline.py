import pytest

import torch

from videosys import OpenSoraConfig, VideoSysEngine
from videosys.models.transformers.cache_model import STDiT3C
from videosys.schedulers.scheduling_si_open_sora import SIRFLOW
from videosys.utils.test import empty_cache


@empty_cache
def test_si(num_gpus):
    config = OpenSoraConfig(num_gpus=num_gpus, use_si=True, si_sparsity=0.5)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # prompt = "In a still frame, a weathered stop sign stands prominently at a quiet intersection, its red paint slightly faded and edges rusted, evoking a sense of time passed. The sign is set against a backdrop of a serene suburban street, lined with tall, leafy trees whose branches gently sway in the breeze. The sky above is a soft gradient of twilight hues, transitioning from deep blue to a warm orange, suggesting the end of a peaceful day. The surrounding area is calm, with neatly trimmed lawns and quaint houses, their windows glowing softly with indoor lights, adding to the tranquil atmosphere."
    # prompt = "A sleek, modern laptop, its screen displaying a vibrant, paused scene, sits on a minimalist wooden desk. The room is bathed in soft, natural light filtering through sheer curtains, casting gentle shadows. The laptop's keyboard is mid-illumination, with a faint glow emanating from the keys, suggesting a moment frozen in time. Dust particles are suspended in the air, caught in the light, adding to the stillness. A steaming cup of coffee beside the laptop remains untouched, with wisps of steam frozen in mid-air. The scene captures a serene, almost magical pause in an otherwise bustling workspace."
    # prompt = "a gorilla eating a carrot"
    # prompt = "a woman is dancing on the top of a skyscraper"
    # prompt = "A person is riding a bike"
    # prompt = "A beautiful coastal beach in spring, waves lapping on sand, Van Gogh style"
    # prompt = "Close up of grapes on a rotating table."
    # prompt = "A panda standing on a surfboard in the ocean in sunset."
    # prompt = "a car on the right of a motorcycle, front view"
    # prompt = "A vintage red phone booth stands alone on a cobblestone street, bathed in the soft glow of a nearby streetlamp. The booth's glass panels reflect the dim light, revealing a glimpse of the old rotary phone inside. Surrounding the booth, ivy climbs up the nearby brick wall, adding a touch of nature to the urban setting. The scene is quiet, with a gentle mist rolling in, creating an air of mystery and nostalgia. The phone booth, a relic of the past, stands as a silent witness to countless stories and conversations, its presence evoking a sense of timelessness."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/si_open_sora_{num_gpus}.mp4")


@empty_cache
def test_base(num_gpus):
    config = OpenSoraConfig(num_gpus=num_gpus)
    engine = VideoSysEngine(config)

    # prompt = "Sunset over the sea."
    # prompt = "In a still frame, a weathered stop sign stands prominently at a quiet intersection, its red paint slightly faded and edges rusted, evoking a sense of time passed. The sign is set against a backdrop of a serene suburban street, lined with tall, leafy trees whose branches gently sway in the breeze. The sky above is a soft gradient of twilight hues, transitioning from deep blue to a warm orange, suggesting the end of a peaceful day. The surrounding area is calm, with neatly trimmed lawns and quaint houses, their windows glowing softly with indoor lights, adding to the tranquil atmosphere."
    # prompt = "A sleek, modern laptop, its screen displaying a vibrant, paused scene, sits on a minimalist wooden desk. The room is bathed in soft, natural light filtering through sheer curtains, casting gentle shadows. The laptop's keyboard is mid-illumination, with a faint glow emanating from the keys, suggesting a moment frozen in time. Dust particles are suspended in the air, caught in the light, adding to the stillness. A steaming cup of coffee beside the laptop remains untouched, with wisps of steam frozen in mid-air. The scene captures a serene, almost magical pause in an otherwise bustling workspace."
    # prompt = "a gorilla eating a carrot"
    # prompt = "A person is riding a bike"
    # prompt = "A beautiful coastal beach in spring, waves lapping on sand, Van Gogh style"
    # prompt = "Close up of grapes on a rotating table."
    # prompt = "A panda standing on a surfboard in the ocean in sunset."
    # prompt = "a car on the right of a motorcycle, front view"
    prompt = "A vintage red phone booth stands alone on a cobblestone street, bathed in the soft glow of a nearby streetlamp. The booth's glass panels reflect the dim light, revealing a glimpse of the old rotary phone inside. Surrounding the booth, ivy climbs up the nearby brick wall, adding a touch of nature to the urban setting. The scene is quiet, with a gentle mist rolling in, creating an air of mystery and nostalgia. The phone booth, a relic of the past, stands as a silent witness to countless stories and conversations, its presence evoking a sense of timelessness."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/base_open_sora_{num_gpus}.mp4")


if __name__ == '__main__':
    test_base(1)
    # test_si(1)
