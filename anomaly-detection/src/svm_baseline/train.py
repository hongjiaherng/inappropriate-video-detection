from typing import Dict
import gc

import torch
import torch.nn as nn
import torch.utils.data
import tqdm.auto as tqdm
import wandb


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
    loss_epoch = 0.0

    model.train()
    with tqdm.tqdm(total=steps_per_epoch, desc="Training", leave=True) as pbar:
        for i, batch in enumerate(train_loader):
            inputs = batch["feature"]  # (B, T, D)
            max_seq_len = torch.max(torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, dim=1)).item()  # max sequence length in the batch
            # Shrink the sequence length as much as possible to the maximum sequence length in the batch
            inputs = inputs[:, :max_seq_len, :].to(device)  # (B, max(T), D)
            labels = batch["binary_target"].view(-1, 1, 1).expand(-1, max_seq_len, -1).to(device)  # (B,) -> (B, T, 1)
            batch_size = inputs.shape[0]
            inputs = inputs.view(batch_size * max_seq_len, -1)  # (B * T, D)
            labels = labels.view(batch_size * max_seq_len, -1)  # (B * T, 1)

            logits = model(inputs)  # (B * T, 1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For logging
            loss = loss.detach().cpu().item()
            loss_epoch += loss

            # Step-level logging (log every n steps)
            if i == 0 or (i + 1) % log_interval_steps == 0:
                wandb.log(
                    {
                        "train/loss_step": loss,
                        "steps_taken": i + 1 + current_epoch * steps_per_epoch,
                    },
                )

            pbar.set_postfix({"loss": loss})
            pbar.update(1)

            # Memory cleanup
            del inputs, labels, logits, loss
            gc.collect()

        # Epoch-level logging (log every epoch)
        loss_epoch /= steps_per_epoch

        wandb.log(
            {
                "train/loss": loss_epoch,
                "epoch": current_epoch + 1,  # at this epoch
                "steps_taken": (current_epoch + 1) * steps_per_epoch,  # total steps taken up to this point
            },
        )
        pbar.set_postfix({"loss": loss_epoch})

    return {"loss": loss_epoch}
