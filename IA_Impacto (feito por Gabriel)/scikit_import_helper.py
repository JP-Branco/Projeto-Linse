""" Módulo com funções para preparação de dados a serem classificados com a biblioteca scikit learn.
"""

import numpy as np
import pandas as pd

def preparar_dados_impacto(df:pd.DataFrame, coluna_picos:str, comprimento_leitura:int, offset_leitura:int=0, altura_pico:int=0, distancia_picos:int=0) -> pd.DataFrame:
    """ Função que encapsula todo o processo de identificar picos, fatiar o data frame e achatá-lo.

    Args:
        df (pd.DataFrame): data frame com os dados a serem tratados.
        coluna_picos (str): coluna do data frame em que se encontram os picos.
        altura_pico (int): altura mínima dos picos a serem encontrados.
        distancia_picos (int): número mínimo de leituras entre um pico e outro.
        offset_leitura (int): número de leituras antes do pico a serem selecionadas.
        comprimento_leitura (int): número de leituras após o pico a serem selecionadas.

    Returns:
        pd.DataFrame: data frame com linhas representando um período de leituras em torno de cada pico.
    """
    
    dados_agrupados_por_pico = agrupar_por_picos(df, coluna_picos, comprimento_leitura, offset_leitura, altura_pico, distancia_picos)

    dfs_achatados = []
    for fatia_de_dados in dados_agrupados_por_pico:
        dfs_achatados.append(achatar_dados(fatia_de_dados))
    
    out_df = pd.concat(dfs_achatados).reset_index().drop('index', axis=1)
    
    return out_df
        


def agrupar_por_picos(df:pd.DataFrame, coluna_alvo:str, comprimento_leitura:int, offset_leitura:int=0, altura_pico:int=0, distancia_picos:int=0) -> list[pd.DataFrame]:
    """ Identifica picos dentro de um data frame e o fatia em torno de cada pico.

    Args:
        df (pd.DataFrame): data frame com os dados a serem tratados.
        coluna_alvo (str): coluna em que estão os picos.
        comprimento_leitura (int): número de leituras após o pico a serem selecionadas.
        offset_leitura (int): número de leituras antes do pico a serem selecionadas.
        altura_pico (int): altura mínima de cada pico.
        distancia_picos (int): número mínimo de leituras entre dois picos.
        
    Returns:
        list[pd.DataFrame]: lista contendo as fatias de dados em torno de cada pico.
    """
    
    from scipy.signal import find_peaks
    
    picos, _ = find_peaks(df[coluna_alvo], height=altura_pico, distance=distancia_picos)

    dados_agrupados_por_pico = []
    for pico in picos:
        dados_agrupados_por_pico.append(df.iloc[pico - offset_leitura:pico + comprimento_leitura])
    
    return dados_agrupados_por_pico



def achatar_dados(df:pd.DataFrame) -> pd.DataFrame:
    """ Achata todos os dados do data frame em uma única linha.
    As labels são numeradas de acordo com a linha dos dados no data frame original.

    Args:
        df (pd.DataFrame): data frame a ser achatado.

    Returns:
        pd.DataFrame: data frame achatado.
    """    
    
    nomes_colunas = []
    for i in range(len(df)):
        nomes_colunas += [f'{nome}_{i}' for nome in df.columns]
    
    dados_achatados = np.array(df).flatten()
    out_df = pd.Series(dados_achatados, index=nomes_colunas).to_frame().T
    
    return out_df