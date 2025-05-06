import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
from ptflops import get_model_complexity_info

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from otta.models.unet import UNet3DConditionModel
from otta.data.dataset import OttaDataset
from otta.pipelines.pipeline_otta import OttaPipeline
from otta.util import save_videos_mp4, ddim_inversion
from einops import rearrange
from otta.models.unet import UNet3DConditionModel

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

import torch.nn.functional as F

#def warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    # frame: [B, C, H, W], flow: [B, 2, H, W]
 #   B, C, H, W = frame.shape
  #  ys = torch.linspace(-1, 1, H, device=frame.device)
   # xs = torch.linspace(-1, 1, W, device=frame.device)
   # grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
  #  base_grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(B, -1, -1, -1)
    # 归一化 flow
   # flow_x = flow[:, 0] / ((W - 1) / 2)
   # flow_y = flow[:, 1] / ((H - 1) / 2)
   # sampling_grid = base_grid + torch.stack((flow_x, flow_y), dim=-1).permute(0,2,3,1)
  #  return F.grid_sample(frame, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True)


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "temp_adapter",
        "lora",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
   # use_prompt_tuning: bool = False, 
  #  prompt_length: int = 24,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

        # —— Prompt Tuning 部 —— 
   # if use_prompt_tuning:
    #    train_pipeline = OttaPipeline(
    #        vae=vae,
   #         text_encoder=text_encoder,
   #         tokenizer=tokenizer,
   #         unet=unet,
     #       scheduler=noise_scheduler,
    #        prompt_length=prompt_length,
     #   )
        # Prompt Tuning: prompt_embeddings
    #    train_pipeline.prompt_embeddings.requires_grad = True

    unet.requires_grad_(False)

    if "lora" in trainable_modules:
        for name, param in unet.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for name, param in text_encoder.named_parameters():
            if "lora" in name:
                param.requires_grad = True
    lora_params = sum(p.numel() for n, p in unet.named_parameters() if "lora" in n and p.requires_grad)
    print(f"[INFO] Trainable LoRA parameters in UNet: {lora_params}")
    
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

  #  if use_prompt_tuning:
    #    optimizer = optimizer_cls(
    #        [train_pipeline.prompt_embeddings],
    #        lr=learning_rate,
    #        betas=(adam_beta1, adam_beta2),
    #        weight_decay=adam_weight_decay,
    #        eps=adam_epsilon,
    #    )
    #else:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

    # Get the training dataset
# Get the training dataset
    train_dataset = OttaDataset(**train_data)


    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

  #  train_dataloader = torch.utils.data.DataLoader(
  #      train_dataset,
  #      batch_size=train_batch_size,
 #       shuffle=True,
 #       num_workers=4,
 #       pin_memory=True,
 #       )

    # Get the validation pipeline
    validation_pipeline = OttaPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
               # encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]
                                # Get the text embedding for conditioning（使用 Prompt Tuning pipeline）
          #      if use_prompt_tuning:
          #          encoder_hidden_states = train_pipeline._encode_prompt(
           #             prompt=batch["prompt"],
          #             device=latents.device,
          #              num_videos_per_prompt=1,
           #             do_classifier_free_guidance=False,
          #              negative_prompt=None,
          #          )
           #     else:
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample  # [B, C, F, H, W]

                loss_spatial = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
             #   loss = loss_spatial
                diff1 = model_pred[:, :, 1:]   - model_pred[:, :, :-1]
                diff2 = model_pred[:, :, 2:]   - model_pred[:, :, :-2]

                zero1 = torch.zeros_like(diff1)
                zero2 = torch.zeros_like(diff2)

                loss_temporal = (F.mse_loss(diff1, zero1, reduction="mean")
                + F.mse_loss(diff2, zero2, reduction="mean")) / 2
                lambda_temporal = 0.05  #  0.01～0.2 
                loss = loss_spatial + lambda_temporal * loss_temporal

             # 4090
          #      pred_frames = validation_pipeline.decode_latents(model_pred).transpose(1, 2)  # [B,F,C,H,W]
               
           #     gt_frames   = batch["pixel_values"]                                       # [B,F,C,H,W]

                #    flow_pred: [B,2,F-1,H,W]
               # flow_model

                # 2.3  t  =warp(t+1)  
             #   warped = []
             #   for t in range(pred_frames.shape[1]-1):
               #     warped.append(warp(pred_frames[:, t], flow_pred[:, :, t]))
              #  warped = torch.stack(warped, dim=1)  # [B,F-1,C,H,W]

              #  with torch.no_grad():
                #    diff    = (pred_frames[:,1:] - warped).abs().mean(dim=2)        # [B,F-1,H,W]
                #    bg_mask = (diff < 0.02).float().unsqueeze(2)                    # [B,F-1,1,H,W]
                
              #  loss_bg = F.l1_loss(warped * bg_mask, pred_frames[:,1:] * bg_mask, reduction="mean")
              #  lambda_bg = 0.1
             #   loss = loss + lambda_bg * loss_bg

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)

                        ddim_inv_latent = None
                        if validation_data.use_inv_latent:
                            inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                            ddim_inv_latent = ddim_inversion(
                                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                num_inv_steps=validation_data.num_inv_steps, prompt=train_data["prompt"])[-1].to(weight_dtype)
                            torch.save(ddim_inv_latent, inv_latents_path)

                                            # eval + no_grad
                    validation_pipeline.unet.eval()
                    validation_pipeline.vae.eval()
                    validation_pipeline.text_encoder.eval()
                    with torch.no_grad():
                        for idx, prompt in enumerate(validation_data.prompts):
                            sample = validation_pipeline(
                                prompt,
                                generator=generator,
                                latents=ddim_inv_latent,
                                **validation_data
                            ).videos
                            save_videos_mp4(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.mp4")
                            samples.append(sample)
                        samples = torch.concat(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}.mp4"
                        save_videos_mp4(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")
                        

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = OttaPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/xzh.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
