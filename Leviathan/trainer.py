""" Add the ability to track metrics and saving checkpoint, together with uploading them to HuggingFace. """

from Leviathan.imports import *
from Leviathan.config import LeviathanModelConfig, training_config
from Leviathan.models import LeviathanComponentModel
from Leviathan.utility import estimate_loss, get_batch
from Leviathan.tokenizer import Tokenizer


config = LeviathanModelConfig(Tokenizer.vocab_size, Tokenizer.max_len)   ## Change this boi.
model = LeviathanComponentModel(config)

model = model.to(training_config.device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

for iter in range(training_config.max_iters):
    #if iter % save_internval == 0
    if iter % training_config.eval_interval == 0 or iter == training_config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save({
            'epoch': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            }, f'model_{iter}_{losses["train"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
