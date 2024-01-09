import logging
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import dataset
import torch
import torch.utils.data
import tqdm.auto as tqdm
import utils
import wandb
from sultani_net.losses import MILRankingLoss
from sultani_net.model import SultaniNet
from sultani_net.test import test_one_epoch
from sultani_net.train import train_one_epoch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_test_metric_val: float,
    ckpt_name: str,
    log_dir: str,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ckpt_path = Path(os.path.join(log_dir, ckpt_name)).as_posix()
    torch.save(
        {
            "epoch": epoch,
            "best_test_metric_val": best_test_metric_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    wandb.save(ckpt_path, base_path=log_dir)

    return ckpt_path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_name: str,
    log_dir: str,
    run_path: str,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    ckpt_path = wandb.restore(ckpt_name, run_path=run_path, root=log_dir).name
    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    best_test_metric_val = ckpt["best_test_metric_val"]

    return model, optimizer, epoch, best_test_metric_val


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
    max_epochs, batch_size, seed, resume_run_id, resume_ckpt_type = (
        training_cfg["max_epochs"],
        training_cfg["batch_size"],
        training_cfg["seed"],  # nullable
        training_cfg["resume_run_id"],  # nullable
        training_cfg["resume_ckpt_type"],  # nullable
    )  # training
    lr, weight_decay = (
        optimizer_cfg["lr"],
        optimizer_cfg["weight_decay"],
    )  # optimizer
    dropout_prob, lambda_smooth, lambda_sparsity = (
        model_cfg["dropout_prob"],
        model_cfg["loss"]["lambda_smooth"],
        model_cfg["loss"]["lambda_sparsity"],
    )  # model
    feature_name, feature_dim, clip_len, sampling_rate, streaming, max_seq_len, num_workers = (
        dataset_cfg["feature_name"],
        dataset_cfg["feature_dim"],
        dataset_cfg["clip_len"],
        dataset_cfg["sampling_rate"],
        dataset_cfg["streaming"],
        dataset_cfg["max_seq_len"],
        dataset_cfg["num_workers"],  # nullable
    )  # dataset
    exp_name, log_interval_steps, ckpt_interval_epochs, test_interval_epochs, log_dir, test_metric = (
        logging_cfg["exp_name"],  # nullable
        logging_cfg["log_interval_steps"],
        logging_cfg["ckpt_interval_epochs"],  # nullable
        logging_cfg["test_interval_epochs"],
        Path(logging_cfg["log_dir"]).as_posix() if logging_cfg["log_dir"] else None,  # nullable
        logging_cfg["test_metric"],
    )  # logging
    project_name = kwargs.get("project_name", None)

    # Some sanity checks
    assert test_metric in ["ap", "rocauc"], "test_metric must be one of ['ap', 'rocauc']."
    assert project_name is not None, "project_name must be specified."
    assert resume_ckpt_type in ["best", "last", None], "resume_ckpt_type must be one of ['best', 'last'] or None."

    os.makedirs(log_dir, exist_ok=True) if log_dir else None

    with wandb.init(
        project=project_name,
        dir=log_dir,
        name=exp_name,
        config={
            "training_cfg": training_cfg,
            "optimizer_cfg": optimizer_cfg,
            "model_cfg": model_cfg,
            "dataset_cfg": dataset_cfg,
            "logging_cfg": logging_cfg,
        },
        id=None if not resume_run_id else resume_run_id,
        resume=None if not resume_run_id else "must",
    ):
        # Log configs, update configs with wandb run name and dir
        exp_name, log_dir = wandb.run.name, Path(wandb.run.dir).as_posix()
        logging_cfg.update({"exp_name": exp_name, "log_dir": log_dir})
        wandb.config.update({"logging_cfg": logging_cfg}, allow_val_change=True)
        logger.info("Running with configs:")
        for k, v in wandb.config.items():
            logger.info(f"{k}: {v}")

        # Seed everything
        utils.seed_everything(seed)

        # Dataloader
        logger.info("Initalizing dataloader...")
        normal_loader = dataset.train_loader(
            config_name=feature_name,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            streaming=streaming,
            num_workers=num_workers,
            filter="normal",
            shuffle=True,
            seed=seed,
        )
        anomaly_loader = dataset.train_loader(
            config_name=feature_name,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            streaming=streaming,
            num_workers=num_workers,
            filter="anomaly",
            shuffle=True,
            seed=seed,
        )
        test_loader = dataset.test_loader(
            config_name=feature_name,
            streaming=streaming,
            num_workers=num_workers,
            clip_len=clip_len,
            sampling_rate=sampling_rate,
        )
        train_steps_per_epoch = min(
            (
                len(normal_loader)
                if not isinstance(normal_loader.dataset, torch.utils.data.IterableDataset)
                else math.ceil(normal_loader.dataset.n_shards * 5 / normal_loader.batch_size)
            ),
            (
                len(anomaly_loader)
                if not isinstance(anomaly_loader.dataset, torch.utils.data.IterableDataset)
                else math.ceil(anomaly_loader.dataset.n_shards * 5 / anomaly_loader.batch_size)
            ),
        )
        test_steps_per_epoch = (
            len(test_loader)
            if not isinstance(test_loader.dataset, torch.utils.data.IterableDataset)
            else math.ceil(test_loader.dataset.n_shards * 5 / test_loader.batch_size)
        )

        # Model, optimizer, scheduler, criterion
        logger.info("Initalizing model, optimizer, scheduler, criterion...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SultaniNet(feature_dim=feature_dim, dropout_prob=dropout_prob).to(device)
        logger.info(f"Model preview: {model}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = MILRankingLoss(lambda_smooth=lambda_smooth, lambda_sparsity=lambda_sparsity, seq_len=max_seq_len)

        start_epoch = 0
        best_test_metric_val = -float("inf")
        if wandb.run.resumed:
            resume_ckpt_type = resume_ckpt_type if resume_ckpt_type else "last"  # default to last
            logger.info(f"Resuming from last checkpoint of {wandb.run.path} at {log_dir} ...")
            model, optimizer, scheduler, start_epoch, best_test_metric_val = load_checkpoint(
                model=model,
                optimizer=optimizer,
                ckpt_name=f"{exp_name}_{resume_ckpt_type}.pth",
                log_dir=log_dir,
                run_path=wandb.run.path,
            )
        else:
            logger.info("Initial evaluation before training...")
            best_test_metric_val = test_one_epoch(
                model=model,
                test_loader=test_loader,
                device=device,
                test_steps_per_epoch=test_steps_per_epoch,
                train_steps_per_epoch=train_steps_per_epoch,
                current_epoch=-1,
                clip_len=clip_len,
                sampling_rate=sampling_rate,
            )[test_metric]

        logger.info("Start training...")
        for epoch in range(start_epoch, max_epochs):
            tqdm.tqdm.write(f"Epoch {epoch+1}/{max_epochs}")
            train_one_epoch(
                model=model,
                criterion=criterion,
                normal_loader=normal_loader,
                anomaly_loader=anomaly_loader,
                optimizer=optimizer,
                device=device,
                steps_per_epoch=train_steps_per_epoch,
                current_epoch=epoch,
                log_interval_steps=log_interval_steps,
            )

            if (epoch + 1) % test_interval_epochs == 0 or epoch == max_epochs - 1 or epoch == 0:
                cur_test_metric_val = test_one_epoch(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    test_steps_per_epoch=test_steps_per_epoch,
                    train_steps_per_epoch=train_steps_per_epoch,
                    current_epoch=epoch,
                    clip_len=clip_len,
                    sampling_rate=sampling_rate,
                )[test_metric]

                # Save the mode checkpoint if it's the best so far
                if cur_test_metric_val >= best_test_metric_val:
                    best_test_metric_val = cur_test_metric_val
                    ckpt_path = save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        best_test_metric_val=best_test_metric_val,
                        ckpt_name=f"{exp_name}_best.pth",
                        log_dir=log_dir,
                    )
                    logger.info(f"Best checkpoint saved to {ckpt_path}.")

            # Checkpoint every n epochs (if specified) or at the last epoch (mandatory)
            if (ckpt_interval_epochs and (epoch + 1) % ckpt_interval_epochs == 0) or epoch == max_epochs - 1 or epoch == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_test_metric_val=best_test_metric_val,
                    ckpt_name=f"{exp_name}_last.pth",
                    log_dir=log_dir,
                )
                logger.info(f"Most recent checkpoint saved to {ckpt_path}.")
