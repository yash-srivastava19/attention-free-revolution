from Leviathan.imports import *
from Leviathan.config import training_config

class MultiHeadedCorrelation(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_correlation(self, Q, K, V, mask=None, dropout=None):
        """ [NEW: This is what is different from the scaled dot attention.] """
        scores = torch.from_numpy(correlate(Q.detach().cpu().numpy(), K.detach().cpu().numpy(), mode='same'))
        scores = scores.to(training_config.device)
        
        scores = scores@K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
    
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
    
        output =  scores@V
        return output
    
    def _different_type_of_correlation(self, Q, K, V, mask=None, dropout=None):
        raise NotImplementedError


    def _another_different_type_of_correlation(self, Q, K, V, mask=None, dropout=None):
        raise NotImplementedError


    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self,x, mask=None):

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        attn_output = self.scaled_correlation(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        
        return output

class Block(nn.Module):
    """ This is where we can allow for creativity. Just that we can go from embed_dim, and back to embed_dim """
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = config.embed_dim

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.corr = MultiHeadedCorrelation(d_model = config.embed_dim, num_heads=config.num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(config.ff_dropout),
        )

    def forward(self, x):
        x = x + self.corr(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x    # This is the x which goes to the decoder. 
    
# Make as many as different block you can, and then we can use those to create different kinds of Blocks and can experiment with them as well !!

class DifferentBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = config.embed_dim

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.corr = MultiHeadedCorrelation(d_model = config.embed_dim, num_heads=config.num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*3),
            nn.ReLU(),
            nn.Linear(embed_dim*3, embed_dim),
            nn.Dropout(config.ff_dropout),
        )

    def forward(self, x):
        """ We can add those, cause MHC and x are of same dimension. """
        x = x + self.corr(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x
    
class AnotherDifferentBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = config.embed_dim

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.corr = MultiHeadedCorrelation(d_model = config.embed_dim, num_heads=config.num_heads)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*6),
            nn.LeakyReLU(),
            nn.Linear(embed_dim*6, embed_dim),
            nn.Dropout(config.ff_dropout),
        )

    def forward(self, x):
        """ We can add those, cause MHC and x are of same dimension. """
        x = x + self.corr(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x
