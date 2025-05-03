import torch
import torch.profiler

from videosys import VideoSysEngine
from videosys.pipelines.open_sora import OpenSoraConfig, OpenSoraPipeline
from videosys.utils.test import empty_cache


@torch.no_grad()
def forward_function(model, latent, embeds, timestep, rot, **kwargs):
    pred = model(
        hidden_states=latent,
        encoder_hidden_states=embeds,
        timestep=timestep,
        image_rotary_emb=rot,
        return_dict=False,
        **kwargs
    )[0]

    return pred


def prof_base_model():
    config = OpenSoraConfig(num_sampling_steps=4)
    engine = VideoSysEngine(config)
    prompt = "A person is petting a cat."

    warmup = 1
    active = 3
    total = warmup + active

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/opensora_base'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(total):
            video = engine.generate(prompt, blur3d=False, verbose=False).video[0]
            print(i, video.shape)
            prof.step()


if __name__ == "__main__":
    prof_base_model()
    # prof_stu_model("THUDM/CogVideoX-5b")
