import pytest
from videosys import OpenSoraConfig, VideoSysEngine
from videosys.utils.test import empty_cache


@empty_cache
def test_stu(prompt, num_infer_steps=32, stu_rate=0.5, measure_flag=False):
    stu_filter = "shortterm"
    config = OpenSoraConfig(
        num_sampling_steps=num_infer_steps,
        enable_ti=True, 
        ti_coef=stu_rate,
        ti_filter=stu_filter,
    )
    
    engine = VideoSysEngine(config)

    if measure_flag:
        # for warming up
        video = engine.generate(prompt, seed=0).video[0]

    video = engine.generate(prompt).video[0]
    
    if not measure_flag:
        output_title = prompt.replace(" ", "_")[:20]
        engine.save_video(video, f"./test_outputs/Opensora/{output_title}_{stu_filter}_{stu_rate}.mp4")


@empty_cache
def test_base(prompt, num_infer_steps=32, measure_flag=False):
    config = OpenSoraConfig(num_sampling_steps=num_infer_steps)
    engine = VideoSysEngine(config)
    
    if measure_flag:
        # for warming up
        video = engine.generate(prompt, seed=0).video[0]
    
    video = engine.generate(prompt, seed=0).video[0]
    
    if not measure_flag:
        output_title = prompt.replace(" ", "_")[:20]
        engine.save_video(video, f"./test_outputs/Opensora/{output_title}_base.mp4")


if __name__ == '__main__':
    # prompt = "Sunset over the sea."
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
    # prompt = "A person is petting an orange cat"
    # prompt = "A person is petting an animal (not a cat)"
    # prompt = "A dramatic, straight-on close-up shot reveals the face of a deep-sea diver inside an old-fashioned brass diving helmet."
    # prompt = "A golden retriever puppy wearing a superhero outfit complete with a mask and cape stands perched on the top of the empire state building in winter, overlooking the city it protects at night."
    # prompt = "A family of grizzly bears sit at a table, dining on salmon sashimi with chopsticks."
    # prompt = "A bird of paradise on a buddha statue."
    # prompt = "An animated claymation video of a plant and his friends."
    # prompt = "A cozy isometric retro futuristic miniature world with spaceships, comets and beautiful planets in the distant background, gorgeously rendered in 8k at maximum quality, tilt shift photography."
    prompt = "A snowy forest landscape with a dirt road running through it. The road is flanked by trees covered in snow, and the ground is also covered in snow. The sun is shining, creating a bright and serene atmosphere. The road appears to be empty, and there are no people or animals visible in the video. The style of the video is a natural landscape shot, with a focus on the beauty of the snowy forest and the peacefulness of the road."
    

    # test_base(prompt)
    # test_stu(prompt)

    steps = [25, 50, 100]
    coefs = [1/2, 1/3, 1/5]

    for step in steps:
        print(f"Testing OpenSora base with {step} steps")
        test_base(prompt, num_infer_steps=step, measure_flag=True)

    for step in steps:
        for coef in coefs:
            print(f"Testing OpenSora STU with {step} steps and {coef} coef")
            test_stu(prompt, num_infer_steps=step, stu_rate=coef, measure_flag=True)
