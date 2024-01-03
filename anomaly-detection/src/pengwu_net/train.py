from typing import Dict
import gc

import torch
import torch.nn as nn
import torch.utils.data
import tqdm.auto as tqdm
import wandb


def debug_progress_bar(steps_per_epoch: int):
    import random
    import time

    with tqdm.tqdm(total=steps_per_epoch, leave=True, desc="Training") as pbar:
        for i in range(steps_per_epoch):
            # Sleep for .5 second
            time.sleep(0.025)
            pbar.set_postfix(
                {
                    "loss": random.random(),
                    "distill": random.random(),
                    "mil_onl": random.random(),
                    "mil_off": random.random(),
                }
            )
            pbar.update(1)

        pbar.set_postfix(
            {
                "loss": random.random(),
                "distill": random.random(),
                "mil_onl": random.random(),
                "mil_off": random.random(),
            }
        )


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps_per_epoch: int,
    current_epoch: int,
    log_interval_steps: int,
) -> Dict[str, float]:
    loss_epoch, distill_epoch, mil_hl_epoch, mil_hlc_epoch = 0.0, 0.0, 0.0, 0.0

    model.train()
    with tqdm.tqdm(total=steps_per_epoch, desc="Training", leave=True) as pbar:
        for i, batch in enumerate(train_loader):
            inputs = batch["feature"]  # (B, T, D)
            labels = batch["binary_target"]
            seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, dim=1)  # (B, 1)

            # Shrink the sequence length as much as possible to the maximum sequence length in the batch
            inputs = inputs[:, : torch.max(seq_len), :]  # (B, max(T), D)
            inputs, labels, seq_len = (
                inputs.to(device),
                labels.to(device),
                seq_len.to(device),
            )

            logits_hl, logits_hlc = model(inputs, seq_len)  # (B, T, 1)
            loss, distill, mil_hl, mil_hlc = criterion(logits_hl, logits_hlc, labels, seq_len)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For logging
            loss = loss.detach().cpu().item()
            distill = distill.detach().cpu().item()
            mil_hl = mil_hl.detach().cpu().item()
            mil_hlc = mil_hlc.detach().cpu().item()

            loss_epoch += loss
            distill_epoch += distill
            mil_hl_epoch += mil_hl
            mil_hlc_epoch += mil_hlc

            # Step-level logging (log every n steps)
            if i == 0 or (i + 1) % log_interval_steps == 0:
                wandb.log(
                    {
                        "train/loss_step": loss,
                        "train/distill_loss_step": distill,
                        "train/mil_loss_offline_step": mil_hl,
                        "train/mil_loss_online_step": mil_hlc,
                        "steps_taken": i + 1 + current_epoch * steps_per_epoch,
                    },
                )

            pbar.set_postfix({"loss": loss, "distill": distill, "mil_off": mil_hl, "mil_onl": mil_hlc})
            pbar.update(1)

            # Memory cleanup
            del inputs, labels, seq_len, logits_hl, logits_hlc, loss, distill, mil_hl, mil_hlc
            gc.collect()

        # Epoch-level logging (log every epoch)
        loss_epoch /= steps_per_epoch
        distill_epoch /= steps_per_epoch
        mil_hl_epoch /= steps_per_epoch
        mil_hlc_epoch /= steps_per_epoch

        wandb.log(
            {
                "train/loss": loss_epoch,
                "train/distill_loss": distill_epoch,
                "train/mil_loss_offline": mil_hl_epoch,
                "train/mil_loss_online": mil_hlc_epoch,
                "epoch": current_epoch + 1,  # at this epoch
                "steps_taken": (current_epoch + 1) * steps_per_epoch,  # total steps taken up to this point
            },
        )
        pbar.set_postfix({"loss": loss_epoch, "distill": distill_epoch, "mil_off": mil_hl_epoch, "mil_onl": mil_hlc_epoch})

    return {
        "loss": loss_epoch,
        "distill": distill_epoch,
        "mil_offline": mil_hl_epoch,
        "mil_online": mil_hlc_epoch,
    }
