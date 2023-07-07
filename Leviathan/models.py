from Leviathan.imports import *
from Leviathan.component_classes import * 
from Leviathan.config import training_config

class LeviathanComponentModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = config.embed_dim
        self.max_len = config.max_len

        self.tok_embed = nn.Embedding(config.vocab_size, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_len, embed_dim))

        self.dropout = nn.Dropout(config.embed_dropout)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)]
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, config.vocab_size)
    
    def forward(self, x, target=None):
        seq_len = x.size(1)
        
        assert seq_len <= self.max_len, f"Sequence Length longer than Model Capacity({seq_len}>{self.max_len})."

        tok_embed = self.tok_embed(x)   # toke_embed.shape = (batch_size, seq_len, embed_dim) 
        pos_embed = self.pos_embed[:, :seq_len, :]  # pos_embed.shape = (1, seq_len, embed_dim) 

        x = self.dropout(tok_embed + pos_embed)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.fc(x)   # x.shape = (batch_size, seq_len, vocab_size)
        
        if target is None:
            loss = None
            
        else:    
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss  # basically, token prediction per position.
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -training_config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class LeviathanModelEnsembling:
    raise NotImplementedError
