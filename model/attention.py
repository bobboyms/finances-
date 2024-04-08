import torch
import torch.nn as nn


class ResidualAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        """
        Initializes the ResidualAttention module.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super(ResidualAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualAttention module.

        Args:
            x (torch.Tensor): The input tensor.
            padding_mask (torch.Tensor): The padding mask tensor.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        attn_output, _ = self.attention(
            x, x, x)  # key_padding_mask=padding_mask.transpose(0, 1)
        return self.norm(x + attn_output)
