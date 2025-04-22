import torch
import torch.profiler

from videosys import CogVideoXConfig, VideoSysEngine
from videosys.pipelines.cogvideox import CogVideoXTauConfig, CogVideoXConfig
from videosys.utils.test import empty_cache


@torch.no_grad()
def forward_function(model, latent, embeds, timestep, rot):
    pred = model(
        hidden_states=latent,
        encoder_hidden_states=embeds,
        timestep=timestep,
        image_rotary_emb=rot,
        return_dict=False,
    )[0]

    return pred


def prof_base_model(model_path):
    config = CogVideoXConfig(
        model_path=model_path, 
        num_gpus=1
    )
    engine = VideoSysEngine(config)

    model = engine.driver_worker.transformer
    dtype = model.patch_embed.proj.weight.dtype
    device = model.patch_embed.proj.weight.device
    print(f"Model dtype: {dtype}, device: {device}")

    fake_latent = torch.randn(2, 13, 16, 60, 90, device=device, dtype=dtype)
    fake_timestep = torch.tensor([1, 1], device=device, dtype=torch.int64)
    fake_embeds = torch.randn(2, 226, 4096, device=device, dtype=dtype)
    
    if model_path.endswith("5b"):
        fake_rot = [
            torch.randn(17550, 64, device=device, dtype=torch.float32),
            torch.randn(17550, 64, device=device, dtype=torch.float32)
        ]
    else:
        fake_rot = None

    warmup = 1
    active = 3
    total = warmup + active

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_path}_base'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(total):
            y = forward_function(model, fake_latent, fake_embeds, fake_timestep, fake_rot)
            print(i, y.shape)
            prof.step()


def prof_stu_model(model_path):
    config = CogVideoXConfig(
        model_path=model_path, 
        num_gpus=1
    )
    engine = VideoSysEngine(config)

    model = engine.driver_worker.transformer
    dtype = model.patch_embed.proj.weight.dtype
    device = model.patch_embed.proj.weight.device
    print(f"Model dtype: {dtype}, device: {device}")

    fake_latent = torch.randn(2, 13, 16, 60, 90, device=device, dtype=dtype)
    fake_timestep = torch.tensor([1, 1], device=device, dtype=torch.int64)
    fake_embeds = torch.randn(2, 226, 4096, device=device, dtype=dtype)
    
    if model_path.endswith("5b"):
        fake_rot = [
            torch.randn(17550, 64, device=device, dtype=torch.float32),
            torch.randn(17550, 64, device=device, dtype=torch.float32)
        ]
    else:
        fake_rot = None

    warmup = 1
    active = 3
    total = warmup + active

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_path}_base'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(total):
            y = forward_function(model, fake_latent, fake_embeds, fake_timestep, fake_rot)
            print(i, y.shape)
            prof.step()


if __name__ == "__main__":
    prof_base_model("THUDM/CogVideoX-2b")
