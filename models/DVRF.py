from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
import torch.optim.adam
from tqdm import tqdm
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps



def lr_hump_beta(k: int, N: int, alpha_max: float,
                 a: float = 3.0, b: float = 6.0) -> float:
    if not (1 <= k <= N):
        raise ValueError("k must be in [1, N]")
    x = (k - 1) / (N - 1)
    pdf = x**(a - 1) * (1 - x)**(b - 1)
    peak = ((a - 1) / (a + b - 2))**(a - 1) * ((b - 1)/(a + b - 2))**(b - 1)
    return alpha_max * pdf / peak

def lr_hump_tail_beta(k: int, N: int, alpha_max: float, beta: float,
                      a: float = 3.0, b: float = 6.0) -> float:
    x = (k - 1) / (N - 1)
    hump = lr_hump_beta(k, N, alpha_max - beta, a, b)      # same shape but reduced amplitude
    tail = beta * x
    return hump + tail

def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward process in flow-matching
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


def calc_v_sd3(pipe, src_tgt_latent_model_input, src_tgt_prompt_embeds, src_tgt_pooled_prompt_embeds, src_guidance_scale, tgt_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tgt_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # ensure latent input dtype matches transformer parameters: ONLY WITH JACOBIAN
        #src_tgt_latent_model_input = src_tgt_latent_model_input.to(next(pipe.transformer.parameters()).dtype)
        # # predict the noise for the source prompt
        noise_pred_src_tgt = pipe.transformer(
            hidden_states=src_tgt_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tgt_prompt_embeds,
            pooled_projections=src_tgt_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tgt_noise_pred_uncond, tgt_noise_pred_text = noise_pred_src_tgt.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tgt = tgt_noise_pred_uncond + tgt_guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)

    return noise_pred_src, noise_pred_tgt


def DVRF_SD3_opt(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tgt_prompt,
    negative_prompt,
    T_steps: int = 50,
    B: int = 1,
    src_guidance_scale: float = 6,
    tgt_guidance_scale: float = 16.5,
    num_steps: int = 50,
    eta: float = 0,
    scheduler_strategy: str = "descending",
    lr: float = 0.02,
    optim: str='SGD',
):
    '''
    DVRF text-to-image optimization for SD3 models
    '''
    zt_edit = x_src.float().clone().requires_grad_(True)  # CHANGE: use FP32 with grad enabled
    if optim == 'SGD':
        if type(lr)==float:
            optimizer = torch.optim.SGD([zt_edit], lr=lr)
        else:
            optimizer = torch.optim.SGD([zt_edit], lr=0.02)

    elif optim == 'SGD_Nesterov':  # SGD with Nesterov momentum
        optimizer = torch.optim.SGD([zt_edit], lr=lr, momentum=0.9, nesterov=True)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop([zt_edit], lr=lr, alpha=0.9)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW([zt_edit], lr=lr)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam([zt_edit], lr=lr)
    else:
        raise ValueError(f'Optimizer {optim} not supported.')
    
    device = x_src.device
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    
    # --- prompt encoding --------------------------------------------------
    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
    
    pipe._guidance_scale = tgt_guidance_scale
    (
        tgt_prompt_embeds,
        tgt_negative_prompt_embeds,
        tgt_pooled_prompt_embeds,
        tgt_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tgt_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
    
    src_tgt_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds, tgt_negative_prompt_embeds, tgt_prompt_embeds], dim=0
    )
    src_tgt_pooled_prompt_embeds = torch.cat(
        [
            src_negative_pooled_prompt_embeds,
            src_pooled_prompt_embeds,
            tgt_negative_pooled_prompt_embeds,
            tgt_pooled_prompt_embeds,
        ],
        dim=0,
    )
    # ----------------------------------------------------------------------
    velocities=[]
    trajectories=[zt_edit.detach().clone()]
    alpha_T_steps=(timesteps[T_steps-2]/1000 - timesteps[T_steps-1] / 1000)/1.6
    alpha_max, beta=alpha_T_steps/1.6, alpha_T_steps/4
    print("alpha_T_steps" , 1.6*alpha_T_steps, )
    if scheduler_strategy == "random":
        for i in range(num_steps):     
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(B):
                ind = torch.randint(2, T_steps - 1, (1,)).item()
                t = timesteps[ind]
                t_i = t / 1000
                print(T_steps)
                alpha_i=2.2*lr_hump_tail_beta(i+1, T_steps+28, alpha_max, beta, a=10, b=8)
                eta_i=eta*i/T_steps
                t_i_FE = timesteps[i] / 1000
                if i + 1 < len(timesteps):
                    t_im1_FE = timesteps[i + 1] / 1000
                else:
                    t_im1_FE = torch.zeros_like(t_i_FE)
                fwd_noise = torch.randn_like(x_src, device=device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tgt = (1 - t_i) * zt_edit + t_i * fwd_noise + eta_i * t_i * (zt_edit - x_src)
                src_tgt_latent_model_input = (
                    torch.cat([zt_src, zt_src, zt_tgt, zt_tgt])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tgt)
                )
                # CHANGE: use inference mode and cast latent input to half precision
                with torch.inference_mode():
                    src_tgt_latent_model_input_fp16 = src_tgt_latent_model_input.half()
                    Vt_src, Vt_tgt = calc_v_sd3(
                        pipe,
                        src_tgt_latent_model_input_fp16,
                        src_tgt_prompt_embeds,
                        src_tgt_pooled_prompt_embeds,
                        src_guidance_scale,
                        tgt_guidance_scale,
                        t,
                    )
                V_delta_avg += (Vt_tgt - Vt_src) / B
            current_lr_FE = t_i_FE - t_im1_FE
            current_lr=alpha_i
            if lr=="FE":
                optimizer.param_groups[0]['lr'] = current_lr_FE
                print("FE lr:", current_lr_FE)
            elif type(lr)== str:
                optimizer.param_groups[0]['lr'] = current_lr
                print("FE lr:", current_lr_FE)
                print(lr, alpha_i)
            velocities.append(V_delta_avg)
            grad=V_delta_avg+(1-eta_i)*(zt_edit-x_src) # test: progressive growing eta
            loss = (zt_edit * grad.detach()).sum()
            velocities.append(V_delta_avg)
            #loss = (zt_edit * V_delta_avg.detach()).sum() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trajectories.append(zt_edit.detach().clone())
    else:  # descending
        for i, t in enumerate(timesteps): # from 0 to T_steps-1, value will go from 1 to 0
            if T_steps - i > num_steps: #i<T_steps-num_steps
                continue
            
            t_i = t / 1000
            if i + 1 < len(timesteps):
                t_im1 = timesteps[i + 1] / 1000
            else:
                t_im1 = torch.zeros_like(t_i)
            print("i, T_steps", i, T_steps)
            alpha_i=2.2*lr_hump_tail_beta(i+1, T_steps+28, alpha_max, beta, a=10, b=8)
            eta_i=eta*i/T_steps
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(B):
                fwd_noise = torch.randn_like(x_src, device=device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tgt = (1 - t_i) * zt_edit + t_i * fwd_noise + eta_i * t_i * (zt_edit - x_src)
                src_tgt_latent_model_input = (
                    torch.cat([zt_src, zt_src, zt_tgt, zt_tgt])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tgt)
                )
                with torch.inference_mode():
                    src_tgt_latent_model_input_fp16 = src_tgt_latent_model_input.half()
                    Vt_src, Vt_tgt = calc_v_sd3(
                        pipe,
                        src_tgt_latent_model_input_fp16,
                        src_tgt_prompt_embeds,
                        src_tgt_pooled_prompt_embeds,
                        src_guidance_scale,
                        tgt_guidance_scale,
                        t,
                    )
                V_delta_avg += (Vt_tgt - Vt_src) / B
            current_lr_FE = t_i - t_im1
            current_lr=alpha_i
            if type(lr)== str:
                optimizer.param_groups[0]['lr'] = current_lr
                print("FE lr:", current_lr_FE)
                print(lr, alpha_i)
            velocities.append(V_delta_avg)
            grad=V_delta_avg+(1-eta_i)*(zt_edit-x_src)
            #loss = 0.5 * grad.pow(2).sum() #equivalent
            loss = (zt_edit * grad.detach()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trajectories.append(zt_edit.detach().clone())
    
    return zt_edit, velocities, trajectories
