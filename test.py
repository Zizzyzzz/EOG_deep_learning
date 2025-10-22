import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x, x, x)
        # Apply normalization and dropout
        x = self.norm1(x + self.dropout1(attn_output))

        # Apply feed-forward network
        ff_output = self.ff(x)
        # Apply normalization and dropout
        x = self.norm2(x + self.dropout2(ff_output))

        return x

if __name__ == '__main__':
    input_dim = 512
    num_heads = 8
    ff_dim = 2048
    dropout = 0.1

    transformer_block = TransformerBlock(input_dim, num_heads, ff_dim, dropout)
    x = torch.randn(32, 128, input_dim)
    output = transformer_block(x)