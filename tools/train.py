#!/usr/bin/env python
# coding=utf-8
import sys
import os
import logging
import math
import shutil
import random
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
import accelerate
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs as DDPK
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from transformers import AutoTokenizer, PretrainedConfig
from PIL import Image
from huggingface_hub import create_repo, upload_folder
from packaging import version

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config_parser import parse_args
from src.utils.util import (
    image_grid, 
    mask_text, 
    import_model_class_from_model_name_or_path, 
    copy_files_to_directory, 
    get_original_py_file
)
from src.utils.align_utils import load_encoders_from_file, preprocess_image, mean_flat

from src.models.SuperModel import SIDModel, InteractNet, LabelEmbedNet, DformerBlock
from src.models.UnetModel import UNet2DConditionNewModel, UNet2DConditionEncodeModel
from src.models.pipeline_sidmodel_img2img import StableDiffusionSIDModelImg2ImgPipeline
from src.data import build_dataset

if is_wandb_available():
    import wandb

check_min_version("0.24.0.dev0")
logger = get_logger(__name__)

def main(args):
    # --------------------------------------------------------------------------
    # 1. Code backup
    # --------------------------------------------------------------------------
    copy_files_to_directory(
        target_dir=args.output_dir,
        current_file_path=__file__,
        extra_files=[
            get_original_py_file(SIDModel),
            get_original_py_file(DformerBlock),
            get_original_py_file(StableDiffusionSIDModelImg2ImgPipeline),
        ],
    )
    
    # --------------------------------------------------------------------------
    # 2. Configuration initialization
    # --------------------------------------------------------------------------
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    kwargs = DDPK(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

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
        
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = torch.Generator(device=accelerator.device)
        
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    # --------------------------------------------------------------------------
    # 3. Model(scheduler) loading and initialization
    # --------------------------------------------------------------------------
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.sd_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.sd_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False)

    def tokenize_captions(captions):
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids

    text_encoder_cls = import_model_class_from_model_name_or_path(args.sd_model_name_or_path, args.revision)

    noise_scheduler = DDIMScheduler.from_pretrained(args.sd_model_name_or_path, subfolder="scheduler")
    noise_scheduler.config.prediction_type = "epsilon"
    
    text_encoder = text_encoder_cls.from_pretrained(args.sd_model_name_or_path, subfolder="text_encoder", revision=args.revision,)
    
    vae = AutoencoderKL.from_pretrained(args.sd_model_name_or_path, subfolder="vae", revision=args.revision)
    unet_ori = UNet2DConditionEncodeModel.from_pretrained(args.sd_model_name_or_path, subfolder="unet", revision=args.revision,)
    unet_gen = UNet2DConditionNewModel.from_pretrained(args.sd_model_name_or_path, subfolder="unet", revision=args.revision,)
    interact_net = InteractNet()
    label_embed_net = LabelEmbedNet.from_pretrained(args.label_embed_dir, revision=args.revision,)
    encoder = None
    encoder_type = None
    if args.enable_alignment:
        encoder, encoder_type, _ = load_encoders_from_file(args.encoder_path, accelerator.device)
    z_dims = [encoder.embed_dim] if encoder is not None else None

    # --------------------------------------------------------------------------
    # 4. Save/Load Hooks
    # --------------------------------------------------------------------------
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "unet_gen"
                    model.unet_gen.save_pretrained(os.path.join(output_dir, sub_dir))
                    sub_dir = "interact_net"
                    model.interact_net.save_pretrained(
                        os.path.join(output_dir, sub_dir)
                    )

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                unet_gen = UNet2DConditionNewModel.from_pretrained(input_dir, subfolder="unet_gen")
                interact_net = InteractNet.from_pretrained(input_dir, subfolder="interact_net")

                model.unet_gen.register_to_config(**unet_gen.config)
                model.interact_net.register_to_config(**interact_net.config)

                model.unet_gen.load_state_dict(unet_gen.state_dict())
                model.interact_net.load_state_dict(interact_net.state_dict())

                del unet_gen, interact_net

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # --------------------------------------------------------------------------
    # 5. Model status configuration
    # --------------------------------------------------------------------------
    label_embed_net.requires_grad_(False)
    vae.requires_grad_(False)
    unet_ori.requires_grad_(False)
    text_encoder.requires_grad_(False)
    interact_net.train()
    unet_gen.train()

    sid_model = SIDModel(
        unet_gen=unet_gen,
        interact_net=interact_net,
        label_embed_net=label_embed_net,
        unet_ori=unet_ori,
        z_dims=z_dims,
    )
    
    # --------------------------------------------------------------------------
    # 6. Other configurations
    # --------------------------------------------------------------------------
    # TODO does not support gradient checkpointing
    if args.gradient_checkpointing:
        sid_model.enable_gradient_checkpointing()

    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(sid_model).dtype != torch.float32:
        raise ValueError(f"Controlnet loaded as datatype {accelerator.unwrap_model(sid_model).dtype}. {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # --------------------------------------------------------------------------
    # 7. Optimizer creation and configurations
    # --------------------------------------------------------------------------
    params_to_optimize = sid_model.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --------------------------------------------------------------------------
    # 8. Dataset configurations
    # --------------------------------------------------------------------------
    if args.dataset is None:
        raise ValueError("`--dataset` is required and must point to a dataset config yaml file.")
    dataset_cfg_file = Path(args.dataset)
    with open(dataset_cfg_file, "r") as f:
        dataset_cfg = yaml.safe_load(f) or {}
    train_dataset = build_dataset(
        dataset_cfg,
        path=args.train_data_dir,
        metadata_file=args.metadata_file,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # --------------------------------------------------------------------------
    # 9. lr_scheduler configurations
    # --------------------------------------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # --------------------------------------------------------------------------
    # 10. Accelerator prepare configurations
    # --------------------------------------------------------------------------
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    sid_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(sid_model, optimizer, train_dataloader, lr_scheduler)

    # --------------------------------------------------------------------------
    # 11. Recalculate steps & Initialize trackers
    # --------------------------------------------------------------------------
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")         # tensorboard cannot handle list types for config
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # --------------------------------------------------------------------------
    # 12. Train!
    # --------------------------------------------------------------------------
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # --------------------------------------------------------------------------
    # 13. Resuming from Checkpoint
    # --------------------------------------------------------------------------
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process,)

    def vae_encode(x):
        img_latents = vae.encode(x.to(dtype=weight_dtype)).latent_dist.sample()
        img_latents = img_latents * vae.config.scaling_factor
        return img_latents
    
    # --------------------------------------------------------------------------
    # 14. Train loop
    # --------------------------------------------------------------------------
    alignment_released = False
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sid_model):
                representators_outputs = None
                if args.enable_alignment and encoder is not None and global_step < args.acc_steps:
                    with torch.no_grad(), accelerator.autocast():
                        representators_output = encoder.forward_features(
                            preprocess_image(batch["image"], encoder_type=encoder_type)
                        )
                        if isinstance(representators_output, dict) and "x_norm_patchtokens" in representators_output:
                            representators_output = representators_output["x_norm_patchtokens"]
                        representators_outputs = [representators_output]
                elif args.enable_alignment and encoder is not None and (not alignment_released) and global_step >= args.acc_steps:
                    if accelerator.is_main_process:
                        accelerator.print("Alignment finished. Release DINO encoder from GPU.")
                    encoder.to("cpu")
                    encoder = None
                    torch.cuda.empty_cache()
                    alignment_released = True

                with accelerator.autocast():
                    img_latents = vae_encode(batch["image"])

                    n, c, h, w = img_latents.shape
                    label_seg = batch["label_index"].unsqueeze(1).long()
                    with torch.no_grad():
                        label_embed = label_embed_net.encode(label_seg)
                    lbl_latents = vae_encode(label_embed)

                    if args.use_rough_guidance:
                        rough_label_seg = batch["rough_label_index"].unsqueeze(1).long()
                        with torch.no_grad():
                            rough_label_embed = label_embed_net.encode(rough_label_seg)
                        rough_lbl_latents = vae_encode(rough_label_embed)
                    else:
                        rough_lbl_latents = torch.zeros_like(lbl_latents)

                    noise = torch.randn_like(lbl_latents)
                    bsz = lbl_latents.shape[0]
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=img_latents.device,
                    )

                    def time_sampling_func(t, T, k):
                        return (1 - (t / T) ** k) * T
                    if args.time_sample_strategy:
                        timesteps = time_sampling_func(timesteps, noise_scheduler.config.num_train_timesteps, k=3)
                    timesteps = torch.clamp(timesteps, max=noise_scheduler.config.num_train_timesteps-1).long()

                    noisy_latents = noise_scheduler.add_noise(lbl_latents, noise, timesteps)
                    batch_mask_text10 = [""] * bsz

                    token_captions10 = torch.stack([example for example in tokenize_captions(batch_mask_text10)])
                    encoder_hidden_states10 = text_encoder(token_captions10.to(accelerator.device))[0]

                    model_pred, representators_tilde = sid_model.get_gen_pred(
                        noise_latents=noisy_latents,
                        cond_latents_1=img_latents,
                        cond_latents_2=rough_lbl_latents,
                        timesteps=timesteps,
                        encoder_hidden_states=encoder_hidden_states10,
                    )

                denoising_loss = F.mse_loss((model_pred).float(), (noise).float(), reduction="mean")

                proj_loss = torch.tensor(0.0, device=accelerator.device)
                if representators_outputs is not None and representators_tilde is not None:
                    for representator_output, representator_tilde in zip(representators_outputs, representators_tilde):
                        if representator_output.shape[1] != representator_tilde.shape[1]:
                            representator_tilde = F.interpolate(
                                representator_tilde.permute(0, 2, 1),
                                size=representator_output.shape[1],
                                mode="linear",
                                align_corners=False,
                            ).permute(0, 2, 1)
                        representator_output = F.normalize(representator_output, dim=-1)
                        representator_tilde = F.normalize(representator_tilde, dim=-1)
                        proj_loss = proj_loss + mean_flat(
                            -(representator_output * representator_tilde).sum(dim=-1)
                        ).mean()
                    proj_loss = proj_loss / len(representators_outputs)
                    total_loss = denoising_loss + proj_loss * args.proj_coeff
                else:
                    total_loss = denoising_loss

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    params_to_clip = sid_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # --------------------------------------------------------------------------
                # 15. Checkpoint Save
                # --------------------------------------------------------------------------
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (len(checkpoints) - args.checkpoints_total_limit + 1)
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": total_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if representators_outputs is not None and representators_tilde is not None:
                logs["proj_loss"] = proj_loss.detach().item()
                logs["denoise_loss"] = denoising_loss.detach().item()
            progress_bar.set_postfix(**logs)
            if global_step % args.log_steps == 0:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # --------------------------------------------------------------------------
    # 16. Create pipeline to save modules
    # --------------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:

        pipeline = StableDiffusionSIDModelImg2ImgPipeline(
            vae=accelerator.unwrap_model(vae),
            tokenizer=tokenizer,
            text_encoder=accelerator.unwrap_model(text_encoder),
            unet_gen=accelerator.unwrap_model(unet_gen),
            unet_ori=accelerator.unwrap_model(unet_ori),
            interact_net=accelerator.unwrap_model(interact_net),
            label_embed_net=accelerator.unwrap_model(label_embed_net),
            scheduler=noise_scheduler,
        )

        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
