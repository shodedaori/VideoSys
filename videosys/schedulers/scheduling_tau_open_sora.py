import math
import torch
import torch.distributed as dist
from tqdm import tqdm

from videosys.models.transformers.os_tau_model import OpenSoraSI
from videosys.schedulers.scheduling_rflow_open_sora import RFLOW, timestep_transform
from videosys.utils.utils import save_obj


class SIRFLOW(RFLOW):
    def __init__(self, coef, filter, *args, **kwargs):
        self.coef = coef
        self.filter = filter
        self.warm_prop = 1 / 3
        self.verbose = False

        if coef >= 1.0: # no sparse in the sampling
            self.warm_prop = 1.0
        
        print(f"Using Tokenwise Inference: coef {coef}, filter {filter}, warm prop {self.warm_prop}")
        super().__init__(*args, **kwargs)
    
    def sample(
        self,
        model,
        z,
        model_args,
        y_null,
        device,
        mask=None,
        guidance_scale=None,
        progress=True,
        verbose=False,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        # text encoding
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, model_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress and dist.get_rank() == 0 else (lambda x: x)

        if self.warm_prop >= 1.0:
            warm_steps = self.num_sampling_steps
        else:
            warm_steps = min(math.ceil(self.num_sampling_steps * self.warm_prop), 10)

        print("Full update warmup steps:", warm_steps)
        si_model = OpenSoraSI(model, self.filter)
        si_model.init_generate_cache(z)
        selec_index = (None, None)

        save_list = []
        for i, t in progress_wrap(list(enumerate(timesteps))):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                x_mask = mask_t_upper.repeat(2, 1)
                
                if torch.all(x_mask):
                    model_args["x_mask"] = None
                else:
                    model_args["x_mask"] = x_mask
                
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)

            # pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            if warm_steps <= i:
                model_args["use_cache"] = True
                model_args["index"] = selec_index
            else:
                model_args["use_cache"] = False
                model_args["index"] = (None, None)

            if self.verbose:
                update_mask = si_model.get_update_mask(z, model_args['index'][1])
            
            output = si_model(z_in, t, **model_args)

            pred = output.chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            selec_index = si_model.index_filter(v_pred, dt, self.coef, sparse_flag=(warm_steps-1<=i<self.num_sampling_steps-1), k=i)

            v_pred = v_pred * dt[:, None, None, None, None]
            z = z + v_pred

            if self.verbose:
                save_list.append((v_pred, z, update_mask))

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        if self.verbose:
            save_obj(save_list, "./exp/", f"coef_{self.coef}_filter_{self.filter}")
        # exit(0)
        return z
