#!/usr/bin/env python
# coding=utf-8
import os
import sys
import logging
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs as DDPK
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.label_embed_config import parse_args
from src.models.SuperModel import LabelEmbedNet

logger = get_logger(__name__)


def _latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not dirs:
        return None
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    return dirs[-1]


def _build_random_labels(batch_size, num_classes, resolution, device):
    return torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size, 1, resolution, resolution),
        device=device,
    )


def main(args):
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

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.init_from:
        label_embed_net = LabelEmbedNet.from_pretrained(args.init_from)
    else:
        label_embed_net = LabelEmbedNet(
            num_classes=args.num_classes,
            label_emb_inchannels=args.label_emb_inchannels,
        )

    if accelerator.unwrap_model(label_embed_net).dtype != torch.float32:
        raise ValueError("LabelEmbedNet must start in float32 precision.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        label_embed_net.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    label_embed_net, optimizer, lr_scheduler = accelerator.prepare(
        label_embed_net, optimizer, lr_scheduler
    )

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    global_step = 0
    if args.resume_from_checkpoint:
        resume_path = None
        if args.resume_from_checkpoint == "latest":
            latest = _latest_checkpoint(args.output_dir)
            if latest:
                resume_path = os.path.join(args.output_dir, latest)
        else:
            resume_path = args.resume_from_checkpoint

        if resume_path and os.path.exists(resume_path):
            accelerator.print(f"Resuming from checkpoint: {resume_path}")
            accelerator.load_state(resume_path)
            global_step = int(os.path.basename(resume_path).split("-")[1])
        else:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.")

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    while global_step < args.max_train_steps:
        with accelerator.accumulate(label_embed_net):
            labels = _build_random_labels(
                args.train_batch_size,
                args.num_classes,
                args.label_resolution,
                accelerator.device,
            )
            with torch.no_grad():
                labels_embed = label_embed_net.encode(labels)

            noise = torch.randn_like(labels_embed) * args.noise_std
            labels_embed = labels_embed + noise
            labels_logits = label_embed_net.decode(labels_embed)

            loss = F.cross_entropy(labels_logits, labels.squeeze(1))
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(label_embed_net.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)

            if accelerator.is_main_process and args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                if args.checkpoints_total_limit is not None:
                    checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    excess = len(checkpoints) - args.checkpoints_total_limit + 1
                    if excess > 0:
                        for ckpt in checkpoints[:excess]:
                            ckpt_path = os.path.join(args.output_dir, ckpt)
                            if os.path.isdir(ckpt_path):
                                shutil.rmtree(ckpt_path)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            if global_step % args.log_steps == 0:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(label_embed_net).save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
