"""Importação, tratamento do CSV e matriz de confusão (parte do Felipe)."""
import pandas as pd
from sklearn.metrics import confusion_matrix

COLUNAS = ["mensagem_id", "classe_real", "probabilidade_phishing"]
ROTULOS = {"phishing": 1, "1": 1, "legitimo": 0, "legítimo": 0, "legitima": 0, "legítima": 0, "0": 0}


def tratar(df: pd.DataFrame) -> pd.DataFrame:
    """Valida e padroniza o CSV: classe_real vira 1 (phishing) / 0 (legítimo)."""
    df = df.rename(columns=str.strip)
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {', '.join(faltando)}")

    # Se a base trouxer a coluna 'conjunto', a avaliação usa só o teste.
    if "conjunto" in df.columns:
        df = df[df["conjunto"].astype(str).str.strip().str.lower() == "teste"]

    df = df.copy()
    df["classe_real"] = df["classe_real"].astype(str).str.strip().str.lower().map(ROTULOS)
    df["probabilidade_phishing"] = pd.to_numeric(
        df["probabilidade_phishing"].astype(str).str.replace(",", "."), errors="coerce"
    )

    invalidas = df["classe_real"].isna() | ~df["probabilidade_phishing"].between(0, 1)
    if invalidas.all():
        raise ValueError("Nenhuma linha válida: verifique classe_real e probabilidade_phishing (0 a 1).")
    df = df[~invalidas]
    df["classe_real"] = df["classe_real"].astype(int)
    df.attrs["descartadas"] = int(invalidas.sum())
    df.attrs["so_teste"] = "conjunto" in df.columns
    return df


def carregar(arquivo) -> pd.DataFrame:
    return tratar(pd.read_csv(arquivo, sep=None, engine="python"))  # aceita , ou ;


def matriz(df: pd.DataFrame, limiar: float) -> dict:
    """Aplica o limiar e devolve VP, FP, FN, VN (classe positiva = phishing)."""
    previsto = (df["probabilidade_phishing"] >= limiar).astype(int)
    (vp, fn), (fp, vn) = confusion_matrix(df["classe_real"], previsto, labels=[1, 0])
    return {"VP": int(vp), "FP": int(fp), "FN": int(fn), "VN": int(vn)}


if __name__ == "__main__":
    teste = pd.DataFrame({
        "mensagem_id": [1, 2, 3, 4, 5, 6],
        "classe_real": ["phishing", "Phishing", "legítimo", "0", "1", "xyz"],
        "probabilidade_phishing": ["0,9", 0.4, 0.6, 0.1, 0.8, 0.5],
    })
    df = tratar(teste)
    assert len(df) == 5 and df.attrs["descartadas"] == 1
    assert matriz(df, 0.5) == {"VP": 2, "FP": 1, "FN": 1, "VN": 1}
    assert matriz(df, 0.3) == {"VP": 3, "FP": 1, "FN": 0, "VN": 1}
    assert matriz(df, 0.95) == {"VP": 0, "FP": 0, "FN": 3, "VN": 2}
    print("ok")
