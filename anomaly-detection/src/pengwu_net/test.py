import gc

import wandb
import tqdm.auto as tqdm
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from metrics import (
    frame_level_prc_auc,
    frame_level_roc_auc,
)


def debug_progress_bar(steps_per_epoch: int):
    import time
    import random

    with tqdm.tqdm(total=steps_per_epoch, leave=True, desc="Testing") as pbar:
        for i in range(steps_per_epoch):
            # Sleep for .5 second
            time.sleep(0.005)
            pbar.set_postfix({"loss": random.random()})
            pbar.update(1)

        pbar.set_postfix(
            {
                "ap_off": random.random(),
                "ap_onl": random.random(),
                "rocauc_off": random.random(),
                "rocauc_onl": random.random(),
                "loss": random.random(),
            }
        )


def test_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    steps_per_epoch: int,
    current_epoch: int,
    clip_len: int,
    sampling_rate: int,
) -> None:
    with torch.no_grad():
        model.eval()
        pred_hlc_full, pred_hl_full, frame_gts_full = [], [], []
        loss_epoch, distill_epoch, mil_hl_epoch, mil_hlc_epoch = 0.0, 0.0, 0.0, 0.0

        with tqdm.tqdm(total=steps_per_epoch, leave=True, desc="Testing") as pbar:
            for i, batch in enumerate(test_loader):
                inputs = batch["feature"].to(device)  # (5, T, D) where 5 is the crop dim
                labels = batch["binary_target"].repeat(5).to(device)  # (1,) -> (5,)
                frame_gts = batch["frame_gts"].numpy()  # (T * clip_len * sampling_rate,)

                logits_hl, logits_hlc = model(inputs, seq_len=None)  # (5, T, 1)

                # Compute loss for logging
                loss, distill, mil_hl, mil_hlc = criterion(logits_hl, logits_hlc, labels, seq_len=None)
                pred_hlc = torch.mean(torch.sigmoid(torch.squeeze(logits_hlc)), dim=0).detach().cpu().numpy()  # Online detection (T,)
                pred_hl = torch.mean(torch.sigmoid(torch.squeeze(logits_hl)), dim=0).detach().cpu().numpy()  # Offline detection (T,)

                pred_hlc_full.append(pred_hlc)
                pred_hl_full.append(pred_hl)
                frame_gts_full.append(frame_gts)

                loss = loss.cpu().item()
                distill = distill.cpu().item()
                mil_hl = mil_hl.cpu().item()
                mil_hlc = mil_hlc.cpu().item()

                loss_epoch += loss
                distill_epoch += distill
                mil_hl_epoch += mil_hl
                mil_hlc_epoch += mil_hlc

                # Progress bar
                pbar.set_postfix({"loss": loss})
                pbar.update(1)

                # Memory cleanup
                del inputs, labels, frame_gts, logits_hl, logits_hlc, loss, distill, mil_hl, mil_hlc
                gc.collect()

            # Compute average loss for logging
            loss_epoch /= steps_per_epoch
            distill_epoch /= steps_per_epoch
            mil_hl_epoch /= steps_per_epoch
            mil_hlc_epoch /= steps_per_epoch

            # Concatenate all batches, expand clip preds to frame preds
            pred_hlc_full = np.concatenate(pred_hlc_full).repeat(sampling_rate * clip_len)  # (N * T,) -> (N * T * clip_len * sampling_rate,)
            pred_hl_full = np.concatenate(pred_hl_full).repeat(sampling_rate * clip_len)
            frame_gts_full = np.concatenate(frame_gts_full)  # (N * T * clip_len * sampling_rate,)

            # Compute metrics and losses for an epoch
            ap_hl, p_hl, r_hl = frame_level_prc_auc(frame_gts_full, pred_hl_full)
            rocauc_hl, fpr_hl, tpr_hl = frame_level_roc_auc(frame_gts_full, pred_hl_full)
            ap_hlc, p_hlc, r_hlc = frame_level_prc_auc(frame_gts_full, pred_hlc_full)
            rocauc_hlc, fpr_hlc, tpr_hlc = frame_level_roc_auc(frame_gts_full, pred_hlc_full)

            # Log metrics and losses for an epoch
            pbar.set_postfix(
                {
                    "ap_off": ap_hl,
                    "ap_onl": ap_hlc,
                    "rocauc_off": rocauc_hl,
                    "rocauc_onl": rocauc_hlc,
                    "loss": loss_epoch,
                }
            )

            wandb.log(
                {
                    "test/ap_offline": ap_hl,
                    "test/ap_online": ap_hlc,
                    "test/rocauc_offline": rocauc_hl,
                    "test/rocauc_online": rocauc_hlc,
                    "test/loss": loss_epoch,
                    "test/distill_loss": distill_epoch,
                    "test/mil_loss_hl": mil_hl_epoch,
                    "test/mil_loss_hlc": mil_hlc_epoch,
                    "epoch": current_epoch,
                },
            )  # TODO: Think about step variable in wandb logging

            # TODO: Plot ROC and PRC curves
            # Plot offline and online on the same ROC figure (slider to slide across epoch)
            # Hover yields the AUC and fpr/tpr at the current epoch
            # wandb.log(
            #     {
            #         "test/roc_offline": wandb.plots.ROC(
            #             y_true=frame_gts_full,
            #             y_probas=pred_hl_full,
            #             classes=["normal", "abnormal"],
            #             title="ROC Offline",
            #         ),
            #         "test/roc_online": wandb.plots.ROC(
            #             y_true=frame_gts_full,
            #             y_probas=pred_hlc_full,
            #             classes=["normal", "abnormal"],
            #             title="ROC Online",
            #         ),
            #     },
            # )

            # Plot offline and online on the same PRC figure (slider to slide across epoch)
            # Hover yields the AUC and precision/recall at the current epoch
