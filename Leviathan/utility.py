""" Maybe we'll move this in the same file as well, or modify the function a lil bit(much more likely). """

from Leviathan.imports import *
from Leviathan.config import training_config, leviathan_model_config

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - training_config.block_size, (training_config.batch_size,))
    x = torch.stack([data[i:i+training_config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+training_config.block_size+1] for i in ix])
    x, y = x.to(training_config.device), y.to(training_config.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(training_config.eval_iters)
        for k in range(training_config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
