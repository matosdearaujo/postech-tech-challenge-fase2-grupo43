"""Configuração central do projeto: caminhos, semente e constantes.

Importe daqui em todos os notebooks para que os resultados sejam reproduzíveis.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Semente fixa: use em TODO ponto com aleatoriedade (split, modelos, CV).
RANDOM_STATE = 42

# Caminhos dos dados
RAW = Path("..") / "data" / "raw" / "application_record.csv"
RAW_2 = Path("..") / "data" / "raw" / "credit_record.csv"
PROCESSED = Path("..") / "data" / "processed"
TARGET = "eh_mau_pagador"

pd.set_option("display.max_columns", None)
