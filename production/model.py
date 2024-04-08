import re
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import ModelXT
from model.device import get_device


def load_json_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Arquivo JSON não encontrado: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON: {file_path}")
        raise


def load_pickle_file(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Arquivo Pickle não encontrado: {file_path}")
        raise
    except pickle.UnpicklingError:
        print(f"Erro ao desempacotar Pickle: {file_path}")
        raise


class Production(nn.Module):
    def __init__(self, device: str, config: dict, state_dict: str, vocab: dict, id_to_label: dict):
        """
        Initializes the Production model.

        Args:
            device (str): The device (e.g., 'cuda', 'mps', 'cpu') to run the model on.
            config (dict): Configuration parameters for the model.
            state_dict (str): Path to the model's state dictionary.
            vocab (dict): A dictionary for word-to-index mapping.
            id_to_label (dict): A dictionary mapping index to label names.
        """
        super(Production, self).__init__()
        self.model = ModelXT(vocab_size=config["vocab_size"], embed_dim=config["embed_dim"],
                             num_heads=config["num_heads"], dropout=config["dropout"],
                             num_classes=config["num_classes"], num_layers=config["num_layers"])
        self.device = device
        self.embed_dim = config["embed_dim"]
        self.model.to(device)
        self.model.load_state_dict(torch.load(state_dict))
        self.model.eval()
        self.id_to_label = id_to_label
        self.vocab = vocab

    def _pad_sequence(self, vectorized_sentences: list) -> torch.Tensor:
        """
        Pad the given sequences of vectorized sentences to a uniform length.

        This method ensures that all sentences have the same length, which is necessary 
        for batch processing in neural networks. It truncates longer sentences to the 
        maximum embed dimension and pads shorter ones with a special padding token.

        Args:
            vectorized_sentences (list): A list of sentences, where each sentence is 
                                    represented as a list of integer tokens.

        Returns:
            torch.Tensor: A tensor of padded and vectorized sentences.
        """
        padded_vectorized_sentences = []
        for seq in vectorized_sentences:
            truncated_seq = seq[:self.embed_dim]
            padded_seq = truncated_seq + \
                [self.vocab["<pad>"]] * (self.embed_dim - len(truncated_seq))
            padded_vectorized_sentences.append(torch.tensor(padded_seq))

        padded_sentences = torch.nn.utils.rnn.pad_sequence(padded_vectorized_sentences,
                                                           batch_first=True, padding_value=self.vocab["<pad>"])
        return padded_sentences

    def _process_sentence(self, sentences: list) -> torch.Tensor:
        # Vetorizar sentenças
        vectorized_sentences = [self.vocab(
            list(sentence)) for sentence in sentences]
        return self._pad_sequence(vectorized_sentences)

    def _get_label(self, indices: torch.Tensor) -> list:
        # Mapear índices para rótulos
        labels = []
        ids = indices.tolist()
        for id in ids:
            label_name = self.id_to_label[id]
            labels.append(label_name)
        return labels

    def forward(self, sentences: list) -> list:
        # Processar sentenças e obter rótulos
        sentences = [re.sub(' +', ' ', sentence.strip().upper())
                     for sentence in sentences]

        with torch.no_grad():
            inputs = self._process_sentence(sentences).to(device=self.device)
            padding_mask = (inputs != 0).to(torch.bool)
            outputs = self.model(inputs, padding_mask)

            probabilities = F.softmax(outputs, dim=1)
            indices = torch.argmax(probabilities, dim=1)
            return self._get_label(indices)


# Carregar recursos externos
config = load_json_file("production/config.json")
vocab = load_pickle_file("production/vocab.pkl")
id_to_label = load_pickle_file("production/id_to_label.pkl")
state_dict_path = "production/model.pth"

production = Production(device=get_device(), config=config,
                        state_dict=state_dict_path, vocab=vocab,
                        id_to_label=id_to_label)
