import torch.optim as optim
from transformers import AdamW


def init_optimizer(model, *args, **params):
    optimizer_type = "adamw"
    learning_rate = 1e-5
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=0)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              weight_decay=0)
    elif optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate,
                             weight_decay=0)
    else:
        raise NotImplementedError

    return optimizer
