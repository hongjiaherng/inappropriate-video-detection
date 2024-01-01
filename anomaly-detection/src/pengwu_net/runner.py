# TODO: Log checkpoint, log ROC and PRC
import math
import logging
import os
from pathlib import Path
from typing import Dict

import dataset
import torch
import torch.utils.data
import tqdm.auto as tqdm
import utils
import wandb
from pengwu_net.losses import PengWuNetLoss
from pengwu_net.model import PengWuNet
from pengwu_net.test import test_one_epoch, debug_progress_bar as debug_test_one_epoch
from pengwu_net.train import train_one_epoch, debug_progress_bar as debug_train_one_epoch


def run(
    training_cfg: Dict,
    optimizer_cfg: Dict,
    model_cfg: Dict,
    dataset_cfg: Dict,
    logging_cfg: Dict,
    logger: logging.Logger,
    **kwargs,
):
    # Unpack configs
    max_epochs, batch_size, seed, pretrained_ckpt = (
        training_cfg["max_epochs"],
        training_cfg["batch_size"],
        training_cfg["seed"],  # nullable
        training_cfg["pretrained_ckpt"],  # nullable
    )  # training
    lr, lr_scheduler_milestones, lr_scheduler_gamma = (
        optimizer_cfg["lr"],
        optimizer_cfg["lr_scheduler"]["milestones"],
        optimizer_cfg["lr_scheduler"]["gamma"],
    )  # optimizer
    dropout_prob, hlc_ctx_len, threshold, sigma, gamma, loss_lambda, loss_is_topk, loss_q = (
        model_cfg["dropout_prob"],
        model_cfg["hlc_ctx_len"],
        model_cfg["threshold"],
        model_cfg["sigma"],
        model_cfg["gamma"],
        model_cfg["loss"]["lambda"],
        model_cfg["loss"]["is_topk"],
        model_cfg["loss"]["q"],
    )  # model
    feature_name, seperated_by_class, feature_dim, clip_len, sampling_rate, streaming, max_seq_len, num_workers = (
        dataset_cfg["feature_name"],
        dataset_cfg["seperated_by_class"],
        dataset_cfg["feature_dim"],
        dataset_cfg["clip_len"],
        dataset_cfg["sampling_rate"],
        dataset_cfg["streaming"],
        dataset_cfg["max_seq_len"],
        dataset_cfg["num_workers"],  # nullable
    )  # dataset
    exp_name, log_every_n_steps, ckpt_every_n_epochs, test_every_n_epochs, log_dir, ckpt_dir = (
        logging_cfg["exp_name"],  # nullable
        logging_cfg["log_every_n_steps"],
        logging_cfg["ckpt_every_n_epochs"],  # nullable
        logging_cfg["test_every_n_epochs"],
        Path(logging_cfg["log_dir"]).as_posix() if logging_cfg["log_dir"] else None,  # nullable
        Path(logging_cfg["ckpt_dir"]).as_posix() if logging_cfg["ckpt_dir"] else None,  # nullable
    )  # logging

    # Some sanity checks
    assert seperated_by_class is False, "PengWuNet does not support constrastive learning."
    os.makedirs(log_dir, exist_ok=True) if log_dir else None
    os.makedirs(ckpt_dir, exist_ok=True) if ckpt_dir else None

    # TODO: Check wandb resume

    with wandb.init(
        project="wsvad",
        name=exp_name,
        config={
            "training_cfg": training_cfg,
            "optimizer_cfg": optimizer_cfg,
            "model_cfg": model_cfg,
            "dataset_cfg": dataset_cfg,
            "logging_cfg": logging_cfg,
        },
        dir=log_dir,
    ):
        # Log configs, update configs with wandb run name and dir
        exp_name, log_dir = wandb.run.name, Path(wandb.run.dir).as_posix()  # Update overrideable logging configs if it got changed
        logging_cfg.update({"exp_name": exp_name, "log_dir": log_dir})
        wandb.config.update({"logging_cfg": logging_cfg}, allow_val_change=True)
        logger.info("Running with configs:")
        for k, v in wandb.config.items():
            logger.info(f"{k}: {v}")

        # Seed everything
        utils.seed_everything(seed)

        # Dataloader
        logger.info("Initalizing dataloader...")
        train_loader = dataset.train_loader(
            config_name=feature_name,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            streaming=streaming,
            filter="all",
            num_workers=num_workers,
            shuffle=True,
            seed=seed,
        )
        test_loader = dataset.test_loader(
            config_name=feature_name,
            streaming=streaming,
            filter="all",
            num_workers=num_workers,
        )
        train_steps_per_epoch = (
            len(train_loader)
            if not isinstance(train_loader.dataset, torch.utils.data.IterableDataset)
            else math.ceil(train_loader.dataset.n_shards * 5 / train_loader.batch_size)
        )
        test_steps_per_epoch = (
            len(test_loader)
            if not isinstance(test_loader.dataset, torch.utils.data.IterableDataset)
            else math.ceil(test_loader.dataset.n_shards * 5 / test_loader.batch_size)
        )

        # Model, optimizer, scheduler, criterion
        logger.info("Initalizing model, optimizer, scheduler, criterion...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PengWuNet(
            feature_dim=feature_dim,
            dropout_prob=dropout_prob,
            hlc_ctx_len=hlc_ctx_len,
            threshold=threshold,
            sigma=sigma,
            gamma=gamma,
        ).to(device)
        logger.info(f"Model preview: {model}")
        optimizer = torch.optim.Adam(
            [
                {"params": model.hl_net.parameters()},
                {"params": model.fuse.parameters()},
                {
                    "params": model.hlc_approx.parameters(),
                    "lr": lr / 2,
                },
            ],
            lr=lr,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            gamma=lr_scheduler_gamma,
            milestones=lr_scheduler_milestones,
        )
        criterion = PengWuNetLoss(lambda_=loss_lambda, is_topk=loss_is_topk, q=loss_q)

        # Load pretrained checkpoint
        # TODO: Check this later
        start_epoch = 0
        if pretrained_ckpt:
            logger.info(f"Loading pretrained checkpoint from {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]

        logger.info("Initial evaluation before training...")
        test_one_epoch(
            model=model,
            criterion=criterion,
            test_loader=test_loader,
            device=device,
            test_steps_per_epoch=test_steps_per_epoch,
            train_steps_per_epoch=train_steps_per_epoch,
            current_epoch=-1,
            clip_len=clip_len,
            sampling_rate=sampling_rate,
        )
        logger.info("Start training...")
        for epoch in range(start_epoch, max_epochs):
            tqdm.tqdm.write(f"Epoch {epoch+1}/{max_epochs}:")
            train_one_epoch(
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                steps_per_epoch=train_steps_per_epoch,
                current_epoch=epoch,
                log_every_n_steps=log_every_n_steps,
            )

            if (epoch + 1) % test_every_n_epochs == 0 or epoch == max_epochs - 1:
                test_one_epoch(
                    model=model,
                    criterion=criterion,
                    test_loader=test_loader,
                    device=device,
                    test_steps_per_epoch=test_steps_per_epoch,
                    train_steps_per_epoch=train_steps_per_epoch,
                    current_epoch=epoch,
                    clip_len=clip_len,
                    sampling_rate=sampling_rate,
                )

            scheduler.step()

            # Checkpoint every n epochs (if specified) or at the last epoch (mandatory)
            if (ckpt_every_n_epochs and (epoch + 1) % ckpt_every_n_epochs == 0) or epoch == max_epochs - 1:
                full_ckpt_dir = os.path.join(ckpt_dir, exp_name)
                ckpt_name = f"{exp_name}_{epoch+1}e.pth"
                os.makedirs(full_ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    os.path.join(full_ckpt_dir, ckpt_name),
                )
            # wandb.save(os.path.join(full_ckpt_dir, ckpt_name))  # TODO: Not sure if this works
