import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.metrics import accuracy
from torch.utils.data import Dataset
from typing import Tuple

# Define a semente de aleatoriedade
torch.manual_seed(42)
np.random.seed(42)


class Trainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, device: torch.device) -> None:
        """
        Inicializa o objeto Trainer.

        Args:
            model (nn.Module): Modelo de rede neural para treinamento.
            train_dataset (Dataset): Conjunto de dados de treinamento.
            test_dataset (Dataset): Conjunto de dados de teste.
            device (torch.device): Dispositivo onde o modelo será treinado (CPU ou GPU).
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.mps_device = device

        # Parâmetros
        self.batch_size = 64
        self.n_epochs_stop = 60

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Inicialização do modelo
        self.model = self.initialize_weights(model)

        # Configuração do Otimizador e Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-3)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.8, patience=3, verbose=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def initialize_weights(self, model: nn.Module) -> nn.Module:
        """
        Inicializa os pesos do modelo usando Kaiming Normal Initialization.

        Args:
        model (nn.Module): Modelo de rede neural.

        Returns:
        nn.Module: Modelo com pesos inicializados.
        """
        for layer in model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                init.kaiming_normal_(
                    layer.weight, mode='fan_in', nonlinearity='relu')
        return model.to(device=self.mps_device)

    def train_epoch(self) -> Tuple[float, float]:
        """
        Realiza o treinamento de uma época.

        Returns:
        Tuple[float, float]: Média de perda e acurácia no conjunto de treinamento.
        """
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        for batch in self.train_loader:
            inputs, labels = batch
            # [0,1,0] Mercado
            # [1,0,0] Restaurante
            # [0,0,1] Saude

            inputs = inputs.to(device=self.mps_device)
            labels = labels.to(device=self.mps_device, dtype=torch.float32)

            padding_mask = (inputs != 0).to(torch.bool)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, padding_mask)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy(outputs, labels)

        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_accuracy / len(self.train_loader)
        return avg_loss, avg_accuracy

    def test_epoch(self) -> Tuple[float, float]:
        """
        Avalia o modelo em uma época no conjunto de teste.

        Returns:
        Tuple[float, float]: Média de perda e acurácia no conjunto de teste.
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        for batch in self.test_loader:
            with torch.no_grad():
                inputs, labels = batch
                inputs = inputs.to(device=self.mps_device)
                labels = labels.to(device=self.mps_device, dtype=torch.float32)

                padding_mask = (inputs != 0).to(torch.bool)

                outputs = self.model(inputs, padding_mask)
                loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()
                total_accuracy += accuracy(outputs, labels)

        avg_loss = total_loss / len(self.test_loader)
        avg_accuracy = total_accuracy / len(self.test_loader)
        return avg_loss, avg_accuracy

    def train(self, num_epochs: int = 500) -> None:
        """
        Executa o processo de treinamento, alternando entre treino e teste, e aplicando parada antecipada se necessário.
        """
        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch()
            test_loss, test_accuracy = self.test_epoch()

            self.scheduler.step(train_loss)

            if epoch % 10 == 0 or epoch == 0:
                print(
                    f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Acc: {train_accuracy}, Test Loss: {test_loss}, Test Acc: {test_accuracy}')

            # Verificação de parada antecipada
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == self.n_epochs_stop:
                print(f'Parada antecipada na época {epoch + 1}')
                break
