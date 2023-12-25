def train_one_epoch(model, train_loader):
    batch = next(iter(train_loader))
    inputs = batch["feature"]
    target = batch["binary_target"]

    print("inputs.shape:", inputs.shape)
    print("target.shape:", target.shape)

    outputs = model(inputs)
