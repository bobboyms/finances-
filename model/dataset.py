import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long)
