from Leviathan.imports import *

""" From this, I think we need to have a directory for testing different types of losses and playing with learning rates as well. """

class BaseLeviathanConfig:
    attn_dropout = 0.1
    embed_dropout = 0.1
    ff_dropout = 0.1

    def __init__(self, vocab_size, max_len, **kwargs) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len

        for key, value in kwargs.items():
            setattr(self, key, value)


class LeviathanModelConfig(BaseLeviathanConfig):
    num_heads = 2
    num_blocks = 2
    embed_dim = 4

leviathan_model_config = LeviathanModelConfig()

class TrainingConfig:
    batch_size = 64         # how many independent sequences will we process in parallel?
    block_size = 512        # what is the maximum context length for predictions?
    max_iters = 5000        # How many epochs the model should be trained for? 
    eval_interval = 500     # After how many intervals should we see the validation loss?
    ckpt_interval = 500     # After how many intervals should we save the model?
    learning_rate = 3e-4    # learning rate.
    eval_iters = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_config = TrainingConfig()
