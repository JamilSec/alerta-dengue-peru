#!/usr/bin/env python3
"""Mide si los dos objetivos posibles del proyecto se pueden aprender con el dataset de dengue.

Uso: verificar_viabilidad.py <dengue.csv>
Evalua con separacion temporal: entrena hasta 2020 y prueba en 2021-2022.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap, f1_score, precision_score, recall_score, roc_auc_score as auc

SEED = 42
CORTE = 2020


def cargar(ruta):
    df = pd.read_csv(ruta, dtype=str, on_bad_lines="skip")
    for c in ["ano", "semana", "edad"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ano", "semana", "edad", "enfermedad", "ubigeo"])
    df = df[df["enfermedad"].str.startswith("DENGUE") & df["semana"].between(1, 53) & df["sexo"].isin(["M", "F"])].copy()
    df["edad_anos"] = df["edad"] / df["tipo_edad"].map({"A": 1, "M": 12, "D": 365})
    df = df.dropna(subset=["edad_anos"])
    df["semana"] = df["semana"].clip(upper=52).astype(int)
    df["ano"] = df["ano"].astype(int)
    return df


def por_ano(df):
    t = pd.crosstab(df["ano"], df["enfermedad"].str.replace("DENGUE ", "", regex=False))
    t["total"] = t.sum(axis=1)
    print("\n== Casos por ano y severidad ==")
    print(t.to_string())


def severidad(df):
    print("\n== A. Severidad (CON senales o GRAVE vs SIN senales) ==")
    d = df.copy()
    d["y"] = (d["enfermedad"] != "DENGUE SIN SEÑALES DE ALARMA").astype(int)
    d["dep"] = d["departamento"].astype("category").cat.codes
    d["sx"] = (d["sexo"] == "M").astype(int)
    d["sen"] = np.sin(2 * np.pi * d["semana"] / 52)
    d["cos"] = np.cos(2 * np.pi * d["semana"] / 52)
    X = ["edad_anos", "sx", "dep", "sen", "cos"]
    tr, te = d[d["ano"] <= CORTE], d[d["ano"] > CORTE]
    print(f"train={len(tr)} prev={tr['y'].mean():.3f} | test={len(te)} prev={te['y'].mean():.3f}")
    m = HGB(max_iter=150, random_state=SEED, categorical_features=[2]).fit(tr[X], tr["y"])
    p = m.predict_proba(te[X])[:, 1]
    print(f"ROC-AUC={auc(te['y'], p):.3f}  PR-AUC={ap(te['y'], p):.3f}  (azar PR-AUC={te['y'].mean():.3f})")


def brotes(df):
    print("\n== B. Brote distrito-semana, una semana antes ==")
    c = df.groupby(["ubigeo", "ano", "semana"]).size().rename("casos").reset_index()
    tot = c.groupby("ubigeo")["casos"].sum()
    distritos = tot[tot >= 50].index
    anos = range(df["ano"].min(), df["ano"].max() + 1)
    idx = pd.MultiIndex.from_product([distritos, anos, range(1, 53)], names=["ubigeo", "ano", "semana"])
    p = c.set_index(["ubigeo", "ano", "semana"]).reindex(idx, fill_value=0).reset_index()
    p = p.sort_values(["ubigeo", "semana", "ano"])
    g = p.groupby(["ubigeo", "semana"])["casos"]
    p["hist_media"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    p["hist_de"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    p["umbral"] = np.maximum(p["hist_media"] + 2 * p["hist_de"], 5)
    p["en_brote"] = (p["casos"] > p["umbral"]).astype(int)

    p = p.sort_values(["ubigeo", "ano", "semana"])
    g = p.groupby("ubigeo")
    for k in [0, 1, 2, 3]:
        p[f"lag{k}"] = g["casos"].shift(k)
    p["media4"] = g["casos"].transform(lambda s: s.rolling(4).mean())
    p["y"] = g["en_brote"].shift(-1)
    p["umbral_sig"] = g["umbral"].shift(-1)
    p["sen"] = np.sin(2 * np.pi * p["semana"] / 52)
    p["cos"] = np.cos(2 * np.pi * p["semana"] / 52)
    X = ["lag0", "lag1", "lag2", "lag3", "media4", "umbral_sig", "sen", "cos"]
    d = p.dropna(subset=X + ["y"]).copy()
    d["y"] = d["y"].astype(int)
    tr, te = d[d["ano"] <= CORTE], d[d["ano"] > CORTE]
    print(f"distritos={len(distritos)} train={len(tr)} prev={tr['y'].mean():.4f} | test={len(te)} prev={te['y'].mean():.4f}")
    m = HGB(max_iter=200, random_state=SEED).fit(tr[X], tr["y"])
    s = m.predict_proba(te[X])[:, 1]
    corte = np.quantile(m.predict_proba(tr[X])[:, 1], 1 - tr["y"].mean())
    yhat = (s >= corte).astype(int)
    base = (te["casos"] > te["umbral"]).astype(int)
    print(f"Modelo: ROC-AUC={auc(te['y'], s):.3f} PR-AUC={ap(te['y'], s):.3f} (azar={te['y'].mean():.4f}) "
          f"P={precision_score(te['y'], yhat, zero_division=0):.3f} R={recall_score(te['y'], yhat):.3f} F1={f1_score(te['y'], yhat):.3f}")
    print(f"Regla persistencia (si hoy hay brote, manana tambien): P={precision_score(te['y'], base, zero_division=0):.3f} "
          f"R={recall_score(te['y'], base):.3f} F1={f1_score(te['y'], base):.3f}")


def main():
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_file():
        sys.exit("Uso: verificar_viabilidad.py <dengue.csv>")
    df = cargar(sys.argv[1])
    print(f"filas validas={len(df)}")
    por_ano(df)
    severidad(df)
    brotes(df)


if __name__ == "__main__":
    main()
