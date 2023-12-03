"""Dummy script for self attention mechanism"""

import torch
import torch.nn as nn


def one_head_attention(x_embed: torch.Tensor, hidden_size: int):
    w_k = nn.Linear(hidden_size, hidden_size)
    w_v = nn.Linear(hidden_size, hidden_size)
    w_q = nn.Linear(hidden_size, hidden_size)

    k = w_k(x_embed)
    v = w_v(x_embed)
    q = w_q(x_embed)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden_size**0.5)

    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, v)
    return output


def multi_head_attention(x_embed: torch.Tensor, hidden_size: int, nb_heads: int):
    head_dim = hidden_size // nb_heads
    w_k = nn.Linear(hidden_size, head_dim)
    w_q = nn.Linear(hidden_size, head_dim)
    w_v = nn.Linear(hidden_size, head_dim)


if __name__ == "__main__":

    # Dummy input representing the input sequence of tokens after embedding + positional encoding
    batch_size = 2
    seq_len = 3
    hidden_size = 4
    # dimension (bach_size, seq_len, hidden_size)
    x_embed = torch.rand(batch_size, seq_len, hidden_size)
    outptut = one_head_attention(x_embed, hidden_size=hidden_size)
    # Compute the k,v,q for each token in the sequence
