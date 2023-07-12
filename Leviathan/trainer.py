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

wandb.watch(model, log_freq = training_config.eval_interval)

optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

for iter in range(training_config.max_iters):
    #if iter % save_internval == 0
    if iter % training_config.eval_interval == 0 or iter == training_config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        wandb.log({'step': iter, 'train_loss': losses['train'], 'val_loss': losses['val']})
        
        torch.save({
            'epoch': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            }, f'model_{iter}_{losses["train"]:.4f}')

        # Add code to upload the model to huggingface.
        api.upload_file(
            path_or_fileobj = f'/path/to/this/model_{i}_{losses["train"]:.4f}',    # TODO: Fix this path accordingly.
            path_in_repo = f'model_{i}_{losses["train"]:.4f}',
            repo_id = "yash-srivastava19/Leviathan",
        )
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
