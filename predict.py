import re
import os
import hashlib
import pandas as pd
from pathlib import Path
from production.model import production
from typing import List, Union


class TransactionProcessor:
    """
    Classe para processar transações financeiras e criar um banco de dados.
    """

    def __init__(self, data_path: str = "rawdata/") -> None:
        """
        Inicializa o processador de transações com o caminho dos dados.

        Args:
            data_path (str): Caminho para os dados brutos.
        """
        self.data_path: Path = Path(data_path)

    def predict_categories(self, sentences: List[str]) -> List[str]:
        """
        Prediz categorias para uma lista de sentenças usando o modelo de produção.

        Args:
            sentences (List[str]): Lista de descrições das transações.

        Returns:
            List[str]: Lista de categorias previstas.
        """
        return production(sentences)

    def convert_price(self, price: Union[str, float]) -> float:
        """
        Converte o preço para um valor numérico.

        Args:
            price (Union[str, float]): Preço a ser convertido.

        Returns:
            float: Preço convertido em formato numérico.
        """
        try:
            if isinstance(price, str):
                clean_price = re.sub(r'[^\d.-]', '', price.replace(',', '.'))
                return float(clean_price)
            return 0.0
        except ValueError:
            return 0.0

    def read_and_process_txt_file(self, file_path: Path, card_type: str) -> pd.DataFrame:
        """
        Lê e processa um arquivo de transações.

        Args:
            file_path (Path): Caminho do arquivo a ser processado.
            card_type (str): Tipo de cartão utilizado na transação.

        Returns:
            pd.DataFrame: DataFrame com as transações processadas.
        """
        df = pd.read_csv(file_path, sep=';', decimal=',',
                         names=['Date', 'Description', 'Price', 'Category'])
        return self.process_file(df, card_type)

    def process_file(self, df, card_type: str):
        df["Card type"] = card_type
        if card_type == "DEBITO":
            df["Type"] = df["Price"].apply(
                lambda x: "DESPESA" if x < 0 else "RECEITA")
        else:
            df["Type"] = df["Price"].apply(
                lambda x: "DESPESA" if x > 0 else "RECEITA")
        df["Price"] = df["Price"].abs()
        df["Hash"] = df.apply(
            lambda row: hashlib.sha256(
                (str(row["Description"]) +
                 str(row["Date"]) + str(row["Price"])).encode()
            ).hexdigest(), axis=1)

        return df

    def read_and_process_xls_file(self, file_path: Path, card_type: str) -> pd.DataFrame:
        """
        Lê e processa um arquivo de transações.

        Args:
            file_path (Path): Caminho do arquivo a ser processado.
            card_type (str): Tipo de cartão utilizado na transação.

        Returns:
            pd.DataFrame: DataFrame com as transações processadas.
        """
        df = pd.read_excel(file_path)
        df = df.drop(columns=['Unnamed: 2'])
        df.rename(columns={'data': 'Date',
                           'lançamento': 'Description',
                           'valor': 'Price'}, inplace=True)

        return self.process_file(df, card_type)

    def get_txt_files(self) -> List[Path]:
        """
        Obtém uma lista de arquivos .txt no caminho de dados.

        Returns:
            List[Path]: Lista de caminhos de arquivos .txt.
        """
        return [file for file in self.data_path.iterdir() if file.suffix == '.txt']

    def get_xls_files(self) -> List[Path]:
        """
        Obtém uma lista de arquivos .xls no caminho de dados.

        Returns:
            List[Path]: Lista de caminhos de arquivos .xls.
        """
        return [file for file in self.data_path.iterdir() if file.suffix == '.xls']

    def create_df(self):
        all_files = []
        for file in self.get_txt_files():
            df = self.read_and_process_txt_file(
                file_path=file, card_type="DEBITO")
            all_files.append(df)

        for file in self.get_xls_files():
            df = self.read_and_process_xls_file(
                file_path=file, card_type="CREDITO")
            all_files.append(df)

        return pd.concat(all_files, ignore_index=True)

    def create_database(self) -> None:
        """
        Cria um banco de dados a partir de arquivos de transações.
        """
        all_df = self.create_df()
        categories = self.predict_categories(all_df["Description"])
        if len(categories) != len(all_df):
            raise ValueError(
                "Número de categorias não corresponde ao número de descrições")

        all_df["Category"] = categories

        try:
            database_df = pd.read_csv("database/data.csv")
        except FileNotFoundError:
            database_df = pd.DataFrame()

        # Filtrando registros com hashes únicos
        unique_hashes = ~all_df['Hash'].isin(database_df['Hash'])
        new_records = all_df[unique_hashes]

        # Adicionando novos registros ao database_df
        database_df = pd.concat([database_df, new_records], ignore_index=True)
        database_df.to_csv("database/data.csv", index=False)


# Exemplo de uso
processor = TransactionProcessor()
processor.create_database()
