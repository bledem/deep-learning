import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(in_features=self.head_dim, out_features=self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, queries, mask):
        # Get number of training examples
        N = queries.shape[0]
        # Get sequence length (`seq_len`), source sentence lenght or target sentence length.
        # Source sentence in the encoder and target sentence in the decoder.
        # Source and target for the key and the query respectively in the encoder-decoder attention.
        value_len, key_len, query_len = (
            values.shape[1],
            keys.shape[1],
            queries.shape[1],
        )  # (N, seq_len, embed_size)

        # Split the Q,K,V matrices into self.heads different pieces
        values = values.reshape(
            N, value_len, self.heads, self.head_dim
        )  # (N, value_len, embed_size) -> (N, seq_len, heads, head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, seq_len, heads, head_dim)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Queries (N, query_len, heads, head_dim) x Keys (N, key_len, heads, head_dim) -> Energy (N, heads, query_len, key_len)
        # `nqhd` means `number of training examples, `q`ueries, nums of `h`eads, head `d`imension
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, seq_len, seq_len)
        # Energy (N, heads, query_len, key_len) x Value (N, val_len, heads, head_dim) -> Attention (N, query_len, heads, head_dim)
        attention = torch.einsum(
            "nhql,nlhd->nqhd", [attention, values]
        )  # since k and v are same, replace key_len and seq_len with `l`.
        # Concatenate heads together
        attention = attention.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(attention)  # does not change the shape of the tensor
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: bool, forward_expansion: int):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        # Like BN but takes the average across the batch and normalize it. LayerNorm takes an average for every single example.
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(values=value, keys=key, queries=query, mask=mask)
        x = self.dropout(self.norm1(attention + query))  # query is always equal to the input for the skip connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        heads,
        device,
        forward_expansion,
        dropout,
        max_lenght,
        num_layers,
    ):
        """
        Encoder for transformer.
        Args:
            src_vocab_size: size of the source vocabulary
            embed_size: embedding size
            heads: number of heads
            device: device to run the model
            forward_expansion: forward expansion
            dropout: dropout rate
            max_lenght: maximum length of the sentence.
            num_layers: number of layers
        """
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_lenght, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape  # batch_size, sequence size
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size=embed_size, heads=heads, dropout=dropout, forward_expansion=forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, query, value, key, src_mask, trg_mask):
        # Decoder self attention
        attention = self.attention(values=query, keys=query, queries=query, mask=trg_mask)
        query = self.dropout(self.norm(attention + query))
        # Encoder-Decoder attention
        out = self.transformer_block(value=value, key=key, query=query, mask=src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_lenght,
    ):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_lenght, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size=embed_size,
                    heads=heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        enc_out,
        src_mask,
        trg_mask,
    ):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(query=x, value=enc_out, key=enc_out, src_mask=src_mask, trg_mask=trg_mask)
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_size: int = 256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_lenght=100,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_lenght=max_lenght,
        )
        self.decoder = Decoder(
            trg_vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_lenght=max_lenght,
        )
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        Make a mask to hide padding and future words.
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        """
        Make a mask to hide padding and future words.
        """
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        # (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(tgt)
        enc_src = self.encoder(x=src, mask=src_mask)
        out = self.decoder(x=tgt, enc_out=enc_src, src_mask=src_mask, trg_mask=trg_mask)
        return out


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=trg_vocab_size,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=trg_vocab_size,
        device=device,
    ).to(device)
    out = model(x, trg[:, :-1])
