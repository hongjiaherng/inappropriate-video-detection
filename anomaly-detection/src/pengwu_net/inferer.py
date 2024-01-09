import logging
import math
from typing import Dict

import dataset
import torch
import torch.utils.data
import wandb
from pengwu_net.model import PengWuNet
from pengwu_net.test import test_no_log


def run(run_path: str, ckpt_type: str, logger: logging.Logger, **kwargs):
    wandb_api = wandb.Api()

    logger.info(f"Loading run {run_path} from wandb ...")
    run = wandb_api.run(run_path)

    logger.info(f"Downloading checkpoint {run.name}_{ckpt_type}.pth from wandb ...")
    ckpt = run.file(f"{run.name}_{ckpt_type}.pth").download(replace=True, root="inference_ckpts")

    logger.info("Loading config ...")
    config = run.config
    for k, v in config.items():
        logger.info(f"{k}: {v}")

    inference(ckpt_path=ckpt.name, logger=logger, **config)


def inference(
    model_cfg: Dict,
    dataset_cfg: Dict,
    ckpt_path: str,
    logger: logging.Logger,
    **kwargs,
):
    feature_name, feature_dim, clip_len, sampling_rate, streaming, num_workers = (
        dataset_cfg["feature_name"],
        dataset_cfg["feature_dim"],
        dataset_cfg["clip_len"],
        dataset_cfg["sampling_rate"],
        dataset_cfg["streaming"],
        dataset_cfg["num_workers"],  # nullable
    )  # dataset

    dropout_prob, hlc_ctx_len, threshold, sigma, gamma = (
        model_cfg["dropout_prob"],
        model_cfg["hlc_ctx_len"],
        model_cfg["threshold"],
        model_cfg["sigma"],
        model_cfg["gamma"],
    )  # model

    logger.info("Initalizing test dataloader...")
    test_loader = dataset.test_loader(
        config_name=feature_name,
        streaming=streaming,
        num_workers=num_workers,
        clip_len=clip_len,
        sampling_rate=sampling_rate,
    )
    test_steps = (
        len(test_loader)
        if not isinstance(test_loader.dataset, torch.utils.data.IterableDataset)
        else math.ceil(test_loader.dataset.n_shards * 5 / test_loader.batch_size)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Initalizing model ...")
    model = PengWuNet(
        feature_dim=feature_dim,
        dropout_prob=dropout_prob,
        hlc_ctx_len=hlc_ctx_len,
        threshold=threshold,
        sigma=sigma,
        gamma=gamma,
    ).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt["epoch"]
    logger.info(f"Loaded checkpoint from {ckpt_path} @ epoch {epoch}")
    logger.info(f"Model preview: {model}")

    result = test_no_log(
        model,
        test_loader,
        device,
        test_steps,
        clip_len=clip_len,
        sampling_rate=sampling_rate,
    )
    ap_offline, ap_online, rocauc_offline, rocauc_online = (
        result["ap_offline"],
        result["ap_online"],
        result["rocauc_offline"],
        result["rocauc_online"],
    )

    logger.info(f"AP (offline): {ap_offline:.4f}")
    logger.info(f"AP (online): {ap_online:.4f}")
    logger.info(f"ROC-AUC (offline): {rocauc_offline:.4f}")
    logger.info(f"ROC-AUC (online): {rocauc_online:.4f}")
