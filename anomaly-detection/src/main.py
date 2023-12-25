from typing import Optional

import torch

import debug_model
import configs
import utils
import pengwu_net


def main(
    max_epochs: int,
    batch_size: int,
    lr: float,
    exp_name: str,
    pretrained_ckpt: str,
    streaming: bool,
    feature_name: str,
    feature_dim: int,
    seed: Optional[int],
    num_workers: Optional[int],
    max_seq_len: int,
    **kwargs,
):
    utils.seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pengwu_net.FullHLNet(feature_dim=feature_dim).to(device)

    debug_model.run(model, device)

    # print("Initializing dataloader...")
    # train_loader = dataset.train_loader(
    #     config_name=feature_name,
    #     batch_size=batch_size,
    #     max_seq_len=max_seq_len,
    #     streaming=streaming,
    #     num_workers=num_workers,
    #     shuffle=True,
    #     seed=seed,
    # )
    # print("Done initializing dataloader.")

    # test_loader = dataset.test_loader(
    #     config_name=feature_name,
    #     streaming=streaming,
    #     num_workers=num_workers,
    # )

    # train.train_one_epoch(model, train_loader)


if __name__ == "__main__":
    args = configs.parse_configs()
    main(**args)
