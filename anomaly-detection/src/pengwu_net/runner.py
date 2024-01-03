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
from pengwu_net.losses import PengWuNetLoss
from pengwu_net.model import PengWuNet
from pengwu_net.test import test_one_epoch
from pengwu_net.train import train_one_epoch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
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
            "scheduler_state_dict": scheduler.state_dict(),
        },
        ckpt_path,
    )
    wandb.save(ckpt_path, base_path=log_dir)

    return ckpt_path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
    ckpt_name: str,
    log_dir: str,
    run_path: str,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.MultiStepLR, int, float]:
    ckpt_path = wandb.restore(ckpt_name, run_path=run_path, root=log_dir).name
    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt["epoch"]
    best_test_metric_val = ckpt["best_test_metric_val"]

    return model, optimizer, scheduler, epoch, best_test_metric_val


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
    max_epochs, batch_size, seed, resume_run_id = (
        training_cfg["max_epochs"],
        training_cfg["batch_size"],
        training_cfg["seed"],  # nullable
        training_cfg["resume_run_id"],  # nullable
    )  # training
    lr, lr_hlc, lr_scheduler_milestones, lr_scheduler_gamma = (
        optimizer_cfg["lr"],
        optimizer_cfg["lr_hlc"],
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
    assert seperated_by_class is False, "PengWuNet does not support constrastive learning."
    assert test_metric in [
        "ap_offline",
        "ap_online",
        "rocauc_offline",
        "rocauc_online",
        "loss",
    ], "test_metric must be one of ['ap_offline', 'ap_online', 'rocauc_offline', 'rocauc_online', 'loss']"
    assert project_name is not None, "project_name must be specified."

    os.makedirs(log_dir, exist_ok=True) if log_dir else None

    with wandb.init(
        project=project_name,
        dir=log_dir,
        name=exp_name if not resume_run_id else None,
        config={
            "training_cfg": training_cfg,
            "optimizer_cfg": optimizer_cfg,
            "model_cfg": model_cfg,
            "dataset_cfg": dataset_cfg,
            "logging_cfg": logging_cfg,
        }
        if not resume_run_id
        else None,
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
        test_loader = dataset.test_loader(config_name=feature_name, streaming=streaming, num_workers=num_workers)
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
                {"params": model.hlc_approx.parameters(), "lr": lr_hlc},
            ],
            lr=lr,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, gamma=lr_scheduler_gamma, milestones=lr_scheduler_milestones)
        criterion = PengWuNetLoss(lambda_=loss_lambda, is_topk=loss_is_topk, q=loss_q)

        start_epoch = 0
        best_test_metric_val = -float("inf") if test_metric != "loss" else float("inf")
        if wandb.run.resumed:
            logger.info(f"Resuming from last checkpoint of {wandb.run.path} at {log_dir} ...")
            model, optimizer, scheduler, start_epoch, best_test_metric_val = load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_name=f"{exp_name}_last.pth",
                log_dir=log_dir,
                run_path=wandb.run.path,
            )
        else:
            logger.info("Initial evaluation before training...")
            best_test_metric_val = test_one_epoch(
                model=model,
                criterion=criterion,
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
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                steps_per_epoch=train_steps_per_epoch,
                current_epoch=epoch,
                log_interval_steps=log_interval_steps,
            )

            if (epoch + 1) % test_interval_epochs == 0 or epoch == max_epochs - 1 or epoch == 0:
                cur_test_metric_val = test_one_epoch(
                    model=model,
                    criterion=criterion,
                    test_loader=test_loader,
                    device=device,
                    test_steps_per_epoch=test_steps_per_epoch,
                    train_steps_per_epoch=train_steps_per_epoch,
                    current_epoch=epoch,
                    clip_len=clip_len,
                    sampling_rate=sampling_rate,
                )[test_metric]

                # Save the mode checkpoint if it's the best so far
                if (test_metric == "loss" and cur_test_metric_val < best_test_metric_val) or (
                    test_metric != "loss" and cur_test_metric_val > best_test_metric_val
                ):
                    best_test_metric_val = cur_test_metric_val
                    ckpt_path = save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        best_test_metric_val=best_test_metric_val,
                        ckpt_name=f"{exp_name}_best.pth",
                        log_dir=log_dir,
                    )
                    logger.info(f"Best checkpoint saved to {ckpt_path}.")

            scheduler.step()

            # Checkpoint every n epochs (if specified) or at the last epoch (mandatory)
            if (ckpt_interval_epochs and (epoch + 1) % ckpt_interval_epochs == 0) or epoch == max_epochs - 1 or epoch == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    best_test_metric_val=best_test_metric_val,
                    ckpt_name=f"{exp_name}_last.pth",
                    log_dir=log_dir,
                )
                logger.info(f"Most recent checkpoint saved to {ckpt_path}.")
