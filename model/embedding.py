import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d: int, n: int = 10000) -> None:
        """
        Initialize the PositionalEncoding module.

        Args:
        seq_len (int): The sequence length.
        d (int): The dimensionality of the positional encoding.
        n (int): The base of the exponential used in the positional encoding calculations.

        Raises:
        ValueError: If d is not an even number.
        """
        super(PositionalEncoding, self).__init__()
        if d % 2 != 0:
            raise ValueError("d deve ser par para codificação posicional.")

        # Pre-calcula os denominadores para eficiência
        denominator = torch.pow(n, 2 * torch.arange(0, d // 2) / d).float()

        # Calcula os índices posicionais para cada dimensão
        pos = torch.arange(seq_len).unsqueeze(1)
        sin_terms = torch.sin(pos / denominator)
        cos_terms = torch.cos(pos / denominator)

        # Intercala os termos seno e cosseno
        pe = torch.zeros(seq_len, d)
        pe[:, 0::2] = sin_terms
        pe[:, 1::2] = cos_terms

        # Adiciona uma dimensão extra para o batch e converte para um buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding module.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The tensor containing the positional encodings.
        """
        return self.pe[:, :x.size(1)]


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, output_dim: int) -> None:
        """
        Initialize the PositionalEmbedding module.

        Args:
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The embedding dimension size for the positional encoding.
        output_dim (int): The output dimension size of the embedding.
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.positional_encoding = PositionalEncoding(
            embed_dim, d=output_dim, n=1000)
        self.word_embedding = nn.Embedding(vocab_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEmbedding module.

        Args:
        x (torch.Tensor): The input tensor of token indices.

        Returns:
        torch.Tensor: The tensor containing combined word and positional embeddings.
        """
        pe = self.positional_encoding(x)
        we = self.word_embedding(x)
        return we + pe


class ExpandValues(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(ExpandValues, self).__init__()
        self.output_size: int = output_size
        self.weights: nn.Parameter = nn.Parameter(torch.normal(
            0, 1, (input_size, output_size)), requires_grad=True)
        self.pe: PositionalEncoding = PositionalEncoding(
            seq_len=input_size, d=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expande o tensor de entrada para facilitar a multiplicação
        x_expanded = x.unsqueeze(-1)

        # Multiplica o tensor expandido pelos pesos
        result = torch.matmul(x_expanded, self.weights)

        return torch.sigmoid(result + self.pe(x))
