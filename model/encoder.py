import torch
import torch.nn as nn
from model.attention import ResidualAttention
from model.feed_forward import ResidualFeedForward
from model.embedding import PositionalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        """
        Initializes the EncoderLayer module.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super(EncoderLayer, self).__init__()

        self.att1 = ResidualAttention(embed_dim=embed_dim,
                                      num_heads=num_heads, dropout=dropout)

        self.ff = ResidualFeedForward(embed_dim=embed_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EncoderLayer module.

        Args:
            x (torch.Tensor): The input tensor.
            padding_mask (torch.Tensor): The padding mask tensor.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        x = self.att1(x, padding_mask)
        return self.ff(x)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, dropout: float, num_layers: int = 3) -> None:
        """
        Initializes the Encoder module.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            num_layers (int): The number of encoder layers. Default is 3.
        """
        super(Encoder, self).__init__()

        self.pe = PositionalEmbedding(vocab_size=vocab_size,
                                      embed_dim=embed_dim, output_dim=embed_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_dim=embed_dim,
                          num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): The input tensor.
            padding_mask (torch.Tensor): The padding mask tensor.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.ff(x)
