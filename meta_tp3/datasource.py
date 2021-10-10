# Esse arquivo tem como objetivo limpar o arquivo de entrada
# A partir dos recursos do notebook em /resources

import pandas as pd
from pathlib import Path


bool_map = {"Sim": True, "Nao": False}
gender_map = {"Masculino": 1, "Feminino": 2}
multiple_lines = {"Sim": 1, "Nao": 2, "SemTelefone": 3}
internet_map = {"Nao": None, "Fibra": 1, "DSL": 2}
online_tools_map = {"Sim": 1, "Nao": 2, "SemInternet": 3}
contract_map = {"Mensal": 1, "Anual": 2, "2 anos": 3}
payment_method_map = {
    "CartaoCredito": 1,
    "DebitoAutomatico": 2,
    "BoletoEletronico": 3,
    "BoletoImpresso": 4,
}


def generate_base_dataframe():
    file_folder = Path(__file__).parent
    file_path = file_folder.joinpath("telecom_users.csv")
    df = pd.read_csv(file_path)

    return df


def filter_dataframe(df):
    df = df.drop("Unnamed: 0", axis=1)
    df = df.dropna(how="all", axis=1)
    df = df.dropna(how="any", axis=0)

    return df


def format_dataframe(df):
    df["TotalGasto"] = pd.to_numeric(df["TotalGasto"], errors="coerce")
    df["IDCliente"] = df["IDCliente"].astype("string")
    df["Aposentado"] = df["Aposentado"].astype("bool")
    df["Casado"] = df["Casado"].map(bool_map).astype("bool")
    df["Dependentes"] = df["Dependentes"].map(bool_map).astype("bool")
    df["Genero"] = df["Genero"].map(gender_map)
    df["ServicoTelefone"] = df["ServicoTelefone"].map(bool_map).astype("bool")
    df["MultiplasLinhas"] = df["MultiplasLinhas"].map(multiple_lines)
    df["ServicoInternet"] = df["ServicoInternet"].map(internet_map)
    df["ServicoSegurancaOnline"] = df["ServicoSegurancaOnline"].map(online_tools_map)
    df["ServicoBackupOnline"] = df["ServicoBackupOnline"].map(online_tools_map)
    df["ProtecaoEquipamento"] = df["ProtecaoEquipamento"].map(online_tools_map)
    df["ServicoSuporteTecnico"] = df["ServicoSuporteTecnico"].map(online_tools_map)
    df["ServicoStreamingTV"] = df["ServicoStreamingTV"].map(online_tools_map)
    df["ServicoFilmes"] = df["ServicoFilmes"].map(online_tools_map)
    df["TipoContrato"] = df["TipoContrato"].map(contract_map)
    df["FaturaDigital"] = df["FaturaDigital"].map(bool_map).astype("bool")
    df["FormaPagamento"] = df["FormaPagamento"].map(payment_method_map)
    df["Churn"] = df["Churn"].map(bool_map).astype("bool")

    return df


def get_dataframe():
    df = generate_base_dataframe()
    df = format_dataframe(df)
    df = filter_dataframe(df)
    return df
