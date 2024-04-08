import torch
import torch.nn as nn
from model.encoder import Encoder


class ModelXT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, dropout: float, num_classes: int, num_layers: int) -> None:
        """
        Initializes the ModelXT module.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            num_classes (int): The number of classes for classification.
            num_layers (int): The number of layers in the encoder.
        """
        super(ModelXT, self).__init__()

        self.encoder = Encoder(vocab_size, embed_dim,
                               num_heads, dropout, num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(embed_dim, embed_dim * 22),
            nn.LayerNorm(embed_dim * 22),
            nn.LeakyReLU(),

            nn.Linear(22 * embed_dim, embed_dim * 16),
            nn.LayerNorm(embed_dim * 16),
            nn.LeakyReLU(),

            nn.Linear(16 * embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ModelXT module.

        Args:
            x (torch.Tensor): The input tensor.
            padding_mask (torch.Tensor): The padding mask tensor.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        x = self.encoder(x, padding_mask)
        # calcula o pooling medio
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
