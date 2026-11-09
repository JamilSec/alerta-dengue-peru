#!/usr/bin/env python3
"""Pipeline completo: alerta temprana de brotes de dengue por distrito-semana (Peru).

Uso: pipeline.py <dengue.csv> <dir_salida>
Salida: metricas.json, tablas CSV, figuras PNG, modelo final y datos para la app Streamlit.
Particion temporal: train <=2020, validacion 2021-2022, prueba 2023-2025.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks, periodogram, savgol_filter
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
UMBRAL_MIN = 5
FIN_TRAIN, FIN_VALID = 2020, 2022
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

R = {}  # resultados para metricas.json


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def guardar(fig, carpeta, nombre):
    fig.savefig(carpeta / f"{nombre}.png")
    plt.close(fig)


# ---------- datos ----------
def cargar(ruta):
    crudo = pd.read_csv(ruta, dtype=str, on_bad_lines="skip")
    n0 = len(crudo)
    nulos = crudo.isna().sum().to_dict()
    df = crudo.copy()
    for c in ["ano", "semana", "edad"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ano", "semana", "edad", "enfermedad", "ubigeo"])
    n1 = len(df)
    df = df[df["enfermedad"].str.startswith("DENGUE") & df["semana"].between(1, 53) & df["sexo"].isin(["M", "F"])].copy()
    n2 = len(df)
    df["edad_anos"] = df["edad"] / df["tipo_edad"].map({"A": 1, "M": 12, "D": 365})
    df = df.dropna(subset=["edad_anos"])
    df = df[df["edad_anos"].between(0, 110)]
    n3 = len(df)
    df = df[(df["ubigeo"] != "999999") & (df["departamento"] != "\\N")].copy()  # ubicacion desconocida
    n4 = len(df)
    df["semana"] = df["semana"].clip(upper=52).astype(int)
    df["ano"] = df["ano"].astype(int)
    df["severidad"] = df["enfermedad"].str.replace("DENGUE ", "", regex=False)
    R["datos"] = {"filas_crudas": n0, "tras_nulos_clave": n1, "tras_validar_categorias": n2, "tras_validar_edad": n3, "sin_ubicacion_conocida": n3 - n4, "filas_finales": n4,
                  "nulos_por_columna": {k: int(v) for k, v in nulos.items()},
                  "columnas": list(crudo.columns), "anos": [int(df.ano.min()), int(df.ano.max())],
                  "departamentos": int(df.departamento.nunique()), "distritos": int(df.ubigeo.nunique()),
                  "severidad": df.severidad.value_counts().to_dict(), "sexo": df.sexo.value_counts().to_dict(),
                  "edad_media": float(df.edad_anos.mean()), "edad_mediana": float(df.edad_anos.median())}
    return crudo, df


FEATS = ["lg_l0", "lg_l1", "lg_l2", "lg_l3", "lg_media4", "lg_max8", "tendencia", "ratio", "en_brote", "brotes4",
         "lg_umbral_sig", "lg_hist_sig", "lg_dep_otros", "dep_brotes", "sen", "cos"]
BASE = ["lg_l0", "lg_l1", "lg_l2", "lg_l3"]


def construir_panel(df):
    c = df.groupby(["ubigeo", "ano", "semana"]).size().rename("casos").reset_index()
    tot = c[c.ano <= FIN_VALID].groupby("ubigeo")["casos"].sum()  # el filtro no mira los anos de prueba
    distritos = tot[tot >= 50].index
    anos = range(df["ano"].min(), df["ano"].max() + 1)
    idx = pd.MultiIndex.from_product([distritos, anos, range(1, 53)], names=["ubigeo", "ano", "semana"])
    p = c.set_index(["ubigeo", "ano", "semana"]).reindex(idx, fill_value=0).reset_index()
    meta = df.groupby("ubigeo")[["departamento", "provincia", "distrito"]].first()
    p = p.join(meta, on="ubigeo")

    p = p.sort_values(["ubigeo", "semana", "ano"]).reset_index(drop=True)
    g = p.groupby(["ubigeo", "semana"])["casos"]
    p["hist_media"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    p["hist_de"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    p["umbral"] = np.maximum(p["hist_media"] + 2 * p["hist_de"], UMBRAL_MIN)
    p["en_brote"] = (p["casos"] > p["umbral"]).astype(int)

    p = p.sort_values(["ubigeo", "ano", "semana"]).reset_index(drop=True)
    g = p.groupby("ubigeo")
    for k in range(4):
        p[f"l{k}"] = g["casos"].shift(k)
    p["media4"] = g["casos"].transform(lambda s: s.rolling(4).mean())
    p["max8"] = g["casos"].transform(lambda s: s.rolling(8).max())
    p["brotes4"] = g["en_brote"].transform(lambda s: s.rolling(4).sum())
    grp = ["departamento", "ano", "semana"]
    p["dep_otros"] = p.groupby(grp)["casos"].transform("sum") - p["casos"]
    p["dep_brotes"] = p.groupby(grp)["en_brote"].transform("sum") - p["en_brote"]
    p["y"] = g["en_brote"].shift(-1)
    p["umbral_sig"] = g["umbral"].shift(-1)
    p["hist_sig"] = g["hist_media"].shift(-1)
    p["sem_sig"] = g["semana"].shift(-1)
    p["casos_sig"] = g["casos"].shift(-1)
    p["ratio"] = (p["casos"] / p["umbral"]).clip(0, 10)
    p["tendencia"] = np.sign(p["l0"] - p["l2"]) * np.log1p(np.abs(p["l0"] - p["l2"]))
    for c_ in ["l0", "l1", "l2", "l3", "media4", "max8", "umbral_sig", "hist_sig", "dep_otros"]:
        p[f"lg_{c_}"] = np.log1p(p[c_])
    p["sen"] = np.sin(2 * np.pi * p["sem_sig"] / 52)
    p["cos"] = np.cos(2 * np.pi * p["sem_sig"] / 52)
    d = p.dropna(subset=FEATS + ["y"]).copy()
    d["y"] = d["y"].astype(int)
    R["panel"] = {"distritos": int(len(distritos)), "umbral_min_casos_distrito": 50, "filas": int(len(d)),
                  "prevalencia": float(d.y.mean()), "n_features": len(FEATS), "features": FEATS}
    return p, d


# ---------- metricas ----------
def corte_optimo(y, s):
    pr, rc, th = precision_recall_curve(y, s)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return float(th[int(np.argmax(f[:-1]))])


def met_bin(d, yhat, s=None):
    y = d["y"].values
    ini = (d["en_brote"].values == 0) & (y == 1)
    m = {"precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0),
         "f1": f1_score(y, yhat, zero_division=0), "balanced_acc": balanced_accuracy_score(y, yhat),
         "recall_inicio_brote": float(yhat[ini].mean()) if ini.any() else None, "n_inicios": int(ini.sum()),
         "n": int(len(y)), "prevalencia": float(y.mean())}
    if s is not None:
        m["pr_auc"] = average_precision_score(y, s)
        m["roc_auc"] = roc_auc_score(y, s)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def puntaje(mod, X):
    return mod.predict_proba(X)[:, 1] if hasattr(mod, "predict_proba") else mod.decision_function(X)


# ---------- sistema experto ----------
REGLAS = [
    ("R1_canal_superado", lambda d: d["casos"] > d["umbral"], "Los casos de esta semana ya superan el canal endemico."),
    ("R2_ascenso_sostenido", lambda d: (d["l0"] > d["l1"]) & (d["l1"] > d["l2"]) & (d["l0"] >= 3) & (d["l0"] >= d["hist_sig"]),
     "Dos semanas seguidas de ascenso y casos sobre la media historica de la proxima semana."),
    ("R3_foco_departamental", lambda d: (d["dep_brotes"] >= 2) & (d["l0"] >= 0.5 * d["umbral_sig"]),
     "Dos o mas distritos del departamento en brote y el distrito cerca de su umbral."),
]


def sistema_experto(d, reglas=REGLAS):
    disparo = pd.Series("", index=d.index, dtype=object)
    alerta = np.zeros(len(d), dtype=bool)
    for nombre, cond, _ in reglas:
        c = cond(d).values
        disparo[c & ~alerta] = nombre
        alerta |= c
    return alerta.astype(int), disparo


# ---------- etapas ----------
def eda(df, p, F):
    a = df.groupby("ano").size()
    fig, ax = plt.subplots(figsize=(7, 3)); a.plot.bar(ax=ax, color="#b5413a"); ax.set(xlabel="Año", ylabel="Casos notificados", title="Casos de dengue notificados por año (Perú)"); guardar(fig, F, "01_casos_por_ano")
    R["eda"] = {"casos_por_ano": {int(k): int(v) for k, v in a.items()}}
    dep = df.departamento.value_counts().head(12)
    R["eda"]["top_departamentos"] = dep.to_dict()
    fig, ax = plt.subplots(figsize=(6, 3.4)); dep.sort_values().plot.barh(ax=ax, color="#3a6ea5"); ax.set(xlabel=f"Casos {df.ano.min()}-{df.ano.max()}", ylabel="", title="Departamentos con más casos"); guardar(fig, F, "02_departamentos")
    h = df.pivot_table(index="ano", columns="semana", values="edad", aggfunc="size", fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); sns.heatmap(np.log1p(h), cmap="YlOrRd", ax=ax, cbar_kws={"label": "log(1+casos)"}); ax.set(xlabel="Semana epidemiológica", ylabel="Año", title="Estacionalidad nacional"); guardar(fig, F, "03_estacionalidad")
    s = pd.crosstab(df.ano, df.severidad, normalize="index")
    fig, ax = plt.subplots(figsize=(7, 3)); s.plot.bar(stacked=True, ax=ax, width=.9, color=["#e0a030", "#b5413a", "#8fb8de"]); ax.set(xlabel="Año", ylabel="Proporción", title="Severidad registrada por año (cambio de clasificación desde 2010)"); ax.legend(fontsize=6, loc="lower left"); guardar(fig, F, "04_severidad_ano")
    fig, ax = plt.subplots(figsize=(6, 2.8)); ax.hist(df.edad_anos, bins=50, color="#3a6ea5"); ax.set(xlabel="Edad (años)", ylabel="Casos", title="Distribución de edad"); guardar(fig, F, "05_edad")
    d0 = df.groupby(["ano", "semana"]).size().rename("c").reset_index()
    R["eda"]["semana_pico_mediana"] = int(d0.loc[d0.groupby("ano")["c"].idxmax(), "semana"].median())
    R["eda"]["brotes_panel_por_ano"] = {int(k): int(v) for k, v in p.groupby("ano")["en_brote"].sum().items()}


def senales(df, F):
    s = df.groupby(["ano", "semana"]).size().reindex(pd.MultiIndex.from_product([range(int(df.ano.min()), int(df.ano.max()) + 1), range(1, 53)]), fill_value=0)
    x = s.values.astype(float)
    t = np.arange(len(x))
    sua = savgol_filter(x, 13, 2)
    pk, _ = find_peaks(sua, distance=26, prominence=np.percentile(sua, 90))
    f, pw = periodogram(x - x.mean(), fs=52)
    dom = f[1:][np.argsort(pw[1:])[::-1][:3]]
    fig, ax = plt.subplots(figsize=(7.5, 2.9)); ax.plot(t, x, lw=.5, color="#aaa", label="Semanal"); ax.plot(t, sua, color="#b5413a", label="Savitzky-Golay"); ax.scatter(pk, sua[pk], color="k", s=12, zorder=3, label="Picos"); ax.set_xticks(np.arange(0, len(x), 104)); ax.set_xticklabels(range(int(df.ano.min()), int(df.ano.max()) + 1, 2)); ax.set(xlabel="Año", ylabel="Casos", title="Serie semanal nacional como señal"); ax.legend(); guardar(fig, F, "06_senal")
    fig, ax = plt.subplots(figsize=(6, 2.6)); ax.semilogy(f[1:], pw[1:], color="#3a6ea5"); ax.set_xlim(0, 6); ax.set(xlabel="Frecuencia (ciclos/año)", ylabel="Potencia", title="Periodograma"); guardar(fig, F, "07_periodograma")
    R["senales"] = {"n_picos": int(len(pk)), "anos_picos": [int(2000 + i // 52) for i in pk], "frecuencias_dominantes_ciclos_ano": [float(v) for v in dom]}


def modelos_supervisados(d, F, salida):
    tr, va, te = d[d.ano <= FIN_TRAIN], d[(d.ano > FIN_TRAIN) & (d.ano <= FIN_VALID)], d[d.ano > FIN_VALID]
    R["particion"] = {k: {"n": int(len(x)), "prevalencia": float(x.y.mean()), "anos": [int(x.ano.min()), int(x.ano.max())]} for k, x in [("train", tr), ("valid", va), ("test", te)]}
    tr_ev = tr.sample(60000, random_state=SEED)
    per = f"{int(te.ano.min())}-{int(te.ano.max())}"
    rng = np.random.RandomState(SEED)
    Xtr, ytr = tr[FEATS], tr["y"]

    knn_idx = rng.choice(len(tr), 40000, replace=False)
    cat = {
        "Dummy (prior)": (DummyClassifier(strategy="prior"), None),
        "Regresión logística": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")), None),
        "KNN (k=15)": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)), knn_idx),
        "SVM lineal": (make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", dual=False, max_iter=3000, random_state=SEED)), None),
        "Árbol de decisión": (DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=SEED), None),
        "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), None),
        "Gradient Boosting": (HGB(max_iter=200, random_state=SEED), None),
        "MLP (64,32)": (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024, early_stopping=True, max_iter=60, random_state=SEED)), None),
    }

    log("grid search Gradient Boosting (pliegues temporales)")
    grid = list(ParameterGrid({"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0], "max_iter": [200]}))
    res_g = []
    for g in grid:
        prs = []
        for a in range(FIN_TRAIN - 2, FIN_TRAIN + 1):
            a_tr, a_va = tr[tr.ano < a], tr[tr.ano == a]
            m = HGB(random_state=SEED, **g).fit(a_tr[FEATS], a_tr.y)
            prs.append(average_precision_score(a_va.y, m.predict_proba(a_va[FEATS])[:, 1]))
        res_g.append({**g, "pr_auc_medio": float(np.mean(prs)), "pr_auc_de": float(np.std(prs))})
    pd.DataFrame(res_g).sort_values("pr_auc_medio", ascending=False).to_csv(salida / "grid_gradient_boosting.csv", index=False)
    mejor = max(res_g, key=lambda x: x["pr_auc_medio"])
    pars = {k: mejor[k] for k in ["learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter"]}
    R["grid"] = {"mejor": mejor, "n_combinaciones": len(grid)}
    cat["Gradient Boosting ajustado"] = (HGB(random_state=SEED, **pars), None)

    filas, ajust, scores = [], {}, {}
    for nom, (m, idx) in cat.items():
        t0 = time.time()
        log(f"entrenando {nom}")
        if idx is None:
            m.fit(Xtr, ytr)
        else:
            m.fit(Xtr.iloc[idx], ytr.iloc[idx])
        sv, ste, str_ = puntaje(m, va[FEATS]), puntaje(m, te[FEATS]), puntaje(m, tr_ev[FEATS])
        if nom.startswith("Dummy"):
            th = 1.0
        else:
            th = corte_optimo(va.y, sv)
        a, b, c = met_bin(va, (sv >= th).astype(int), sv), met_bin(te, (ste >= th).astype(int), ste), met_bin(tr_ev, (str_ >= th).astype(int), str_)
        filas.append({"modelo": nom, "umbral": th, "train_f1": c["f1"], "train_pr_auc": c["pr_auc"], "valid_pr_auc": a["pr_auc"], "valid_f1": a["f1"],
                      "test_pr_auc": b["pr_auc"], "test_roc_auc": b["roc_auc"], "test_precision": b["precision"], "test_recall": b["recall"], "test_f1": b["f1"], "test_bal_acc": b["balanced_acc"], "test_recall_inicio": b["recall_inicio_brote"], "seg": time.time() - t0})
        ajust[nom], scores[nom] = m, ste

    for nom, sub in [("Regla: persistencia (R1)", REGLAS[:1]), ("Sistema experto (R1-R3)", REGLAS)]:
        yv, _ = sistema_experto(va, sub); yt, _ = sistema_experto(te, sub); ytr_, _ = sistema_experto(tr_ev, sub)
        a, b, c = met_bin(va, yv), met_bin(te, yt), met_bin(tr_ev, ytr_)
        filas.append({"modelo": nom, "umbral": None, "train_f1": c["f1"], "train_pr_auc": None, "valid_pr_auc": None, "valid_f1": a["f1"], "test_pr_auc": None, "test_roc_auc": None,
                      "test_precision": b["precision"], "test_recall": b["recall"], "test_f1": b["f1"], "test_bal_acc": b["balanced_acc"], "test_recall_inicio": b["recall_inicio_brote"], "seg": 0.0})
    tab = pd.DataFrame(filas)
    tab.to_csv(salida / "comparacion_modelos.csv", index=False)
    R["comparacion"] = tab.to_dict(orient="records")
    _, dis = sistema_experto(te)
    R["reglas_disparos_test"] = dis.value_counts().to_dict()

    ml = tab[tab.valid_pr_auc.notna() & ~tab.modelo.str.startswith("Dummy")]
    elegido = ml.sort_values("valid_pr_auc", ascending=False).iloc[0]["modelo"]
    th = float(tab.loc[tab.modelo == elegido, "umbral"].iloc[0])
    R["seleccion"] = {"modelo": elegido, "criterio": f"mayor PR-AUC en validacion {FIN_TRAIN + 1}-{FIN_VALID}", "umbral_validacion": th}
    log(f"elegido: {elegido}")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for nom in ["Regresión logística", "Random Forest", "Gradient Boosting ajustado", "MLP (64,32)"]:
        pr, rc, _ = precision_recall_curve(te.y, scores[nom]); ax.plot(rc, pr, label=f"{nom} (AP={average_precision_score(te.y, scores[nom]):.3f})", lw=1.1)
    ax.axhline(te.y.mean(), ls="--", color="grey", label="Azar"); ax.set(xlabel="Recall", ylabel="Precisión", title=f"Curvas precisión-recall (prueba {per})"); ax.legend(fontsize=7); guardar(fig, F, "08_curvas_pr")
    t2 = tab[~tab.modelo.str.startswith("Dummy")].sort_values("test_f1")
    fig, ax = plt.subplots(figsize=(6.5, 3.5)); ax.barh(t2.modelo, t2.test_f1, color=["#b5413a" if m.startswith(("Regla", "Sistema")) else "#3a6ea5" for m in t2.modelo]); ax.set(xlabel="F1 en prueba", title="Comparación de modelos"); guardar(fig, F, "09_comparacion")

    # modelo final: ventana completa hasta 2020, umbral fijado en validacion
    fin = clone(cat[elegido][0])
    trva = d[d.ano <= FIN_VALID]
    fin.fit(trva[FEATS], trva.y)
    s = puntaje(fin, te[FEATS])
    yhat = (s >= th).astype(int)
    R["final"] = met_bin(te, yhat, s)
    R["final"]["matriz_confusion"] = confusion_matrix(te.y, yhat).tolist()
    s_t = puntaje(fin, tr_ev[FEATS]); R["final"]["train_f1_ref"] = met_bin(tr_ev, (s_t >= th).astype(int), s_t)["f1"]
    joblib.dump({"modelo": fin, "umbral": th, "features": FEATS}, salida / "modelo_final.joblib")
    cm = confusion_matrix(te.y, yhat)
    fig, ax = plt.subplots(figsize=(3.4, 2.9)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["No brote", "Brote"], yticklabels=["No brote", "Brote"]); ax.set(xlabel="Predicho", ylabel="Real", title="Matriz de confusión (prueba)"); guardar(fig, F, "10_matriz_confusion")

    log("importancia por permutacion")
    ms = te.sample(30000, random_state=SEED)
    pi = permutation_importance(fin, ms[FEATS], ms.y, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=FEATS).sort_values()
    R["importancia"] = imp.sort_values(ascending=False).to_dict()
    fig, ax = plt.subplots(figsize=(5.5, 3.6)); imp.plot.barh(ax=ax, color="#3a6ea5"); ax.set(xlabel="Caída de PR-AUC al permutar", title="Importancia de variables"); guardar(fig, F, "11_importancia")

    t = te.assign(p=s, yhat=yhat)
    e = t.groupby("departamento").apply(lambda g: pd.Series({"filas": len(g), "brotes": int(g.y.sum()), "VP": int(((g.y == 1) & (g.yhat == 1)).sum()), "FP": int(((g.y == 0) & (g.yhat == 1)).sum()), "FN": int(((g.y == 1) & (g.yhat == 0)).sum())}), include_groups=False)
    e["recall"] = e.VP / e.brotes.replace(0, np.nan); e["precision"] = e.VP / (e.VP + e.FP).replace(0, np.nan)
    e.sort_values("brotes", ascending=False).to_csv(salida / "errores_departamento.csv")
    R["errores_departamento"] = e.sort_values("brotes", ascending=False).head(8).reset_index().to_dict(orient="records")
    fn = t[(t.y == 1) & (t.yhat == 0)]
    R["analisis_errores"] = {"FN_media_casos_semana_actual": float(fn.casos.mean()), "VP_media_casos_semana_actual": float(t[(t.y == 1) & (t.yhat == 1)].casos.mean()),
                             "FN_que_son_inicio": float((fn.en_brote == 0).mean()), "FP_que_estaban_en_brote": float((t[(t.y == 0) & (t.yhat == 1)].en_brote == 1).mean())}

    top = t.groupby("ubigeo").y.sum().idxmax()
    g = t[t.ubigeo == top].sort_values(["ano", "semana"])
    fig, ax = plt.subplots(figsize=(7.5, 2.9)); x = np.arange(len(g)); ax.plot(x, g.casos_sig, color="#333", label="Casos (semana siguiente)"); ax.plot(x, g.umbral_sig, color="#e0a030", label="Umbral"); ax.scatter(x[g.yhat == 1], g.casos_sig[g.yhat == 1], color="#b5413a", s=12, label="Alerta del modelo", zorder=3); ax.set(title=f"Ejemplo: {g.distrito.iloc[0]} ({g.departamento.iloc[0]}), {per}", xlabel="Semana del periodo", ylabel="Casos"); ax.legend(fontsize=7); guardar(fig, F, "12_ejemplo_distrito")

    log("validacion temporal expansiva")
    filas_cv = []
    for nom in [n for n in cat if not n.startswith("Dummy")]:
        for a in range(FIN_VALID - 6, FIN_VALID + 1):
            a_tr, a_te = d[d.ano < a], d[d.ano == a]
            if nom.startswith("KNN") and len(a_tr) > 40000:
                a_tr = a_tr.sample(40000, random_state=SEED)
            m = clone(cat[nom][0]).fit(a_tr[FEATS], a_tr.y)
            sc = puntaje(m, a_te[FEATS])
            filas_cv.append({"modelo": nom, "ano": a, "pr_auc": average_precision_score(a_te.y, sc), "prevalencia": float(a_te.y.mean())})
    cv = pd.DataFrame(filas_cv); cv.to_csv(salida / "validacion_temporal.csv", index=False)
    R["validacion_temporal"] = {k: {"media": float(v.pr_auc.mean()), "de": float(v.pr_auc.std()), "min": float(v.pr_auc.min()), "max": float(v.pr_auc.max())} for k, v in cv.groupby("modelo")}

    log("experimentos de mejora")
    ba = HGB(random_state=SEED, **pars).fit(tr[BASE], tr.y); sb = ba.predict_proba(te[BASE])[:, 1]; tb = corte_optimo(va.y, ba.predict_proba(va[BASE])[:, 1])
    hgb = ajust["Gradient Boosting ajustado"]; sf = hgb.predict_proba(te[FEATS])[:, 1]; tf = float(tab.loc[tab.modelo == "Gradient Boosting ajustado", "umbral"].iloc[0])
    idx_a, idx_b = train_test_split(d.index, test_size=.2, random_state=SEED, stratify=d.y)
    ra = HGB(random_state=SEED, **pars).fit(d.loc[idx_a, FEATS], d.loc[idx_a, "y"]); sra = ra.predict_proba(d.loc[idx_b, FEATS])[:, 1]
    R["mejoras"] = {
        "umbral_05": met_bin(te, (sf >= .5).astype(int), sf), "umbral_ajustado": met_bin(te, (sf >= tf).astype(int), sf), "umbral_valor": tf,
        "solo_rezagos": met_bin(te, (sb >= tb).astype(int), sb), "todas_variables": met_bin(te, (sf >= tf).astype(int), sf),
        "particion_aleatoria_pr_auc": float(average_precision_score(d.loc[idx_b, "y"], sra)), "particion_temporal_pr_auc": float(average_precision_score(te.y, sf))}

    log("seleccion de variables por busqueda voraz")
    sub = tr.sample(60000, random_state=SEED)
    sel, resto, hist, mejor_p = [], list(FEATS), [], 0.0
    while resto:
        cand = [(average_precision_score(va.y, HGB(max_iter=60, random_state=SEED).fit(sub[sel + [c]], sub.y).predict_proba(va[sel + [c]])[:, 1]), c) for c in resto]
        p_, c_ = max(cand)
        if p_ - mejor_p < 0.002:
            break
        sel.append(c_); resto.remove(c_); mejor_p = p_; hist.append({"variable": c_, "pr_auc_valid": float(p_)})
    R["busqueda_variables"] = hist

    mlp = ajust["MLP (64,32)"].steps[-1][1]
    fig, (ax, a2) = plt.subplots(1, 2, figsize=(7, 2.6)); ax.plot(mlp.loss_curve_, color="#3a6ea5"); ax.set(xlabel="Época", ylabel="Pérdida", title="Pérdida en entrenamiento"); a2.plot(mlp.validation_scores_, color="#3a6ea5"); a2.set(xlabel="Época", ylabel="Exactitud", title="Validación interna"); fig.tight_layout(); guardar(fig, F, "13_mlp"); R["mlp"] = {"epocas": len(mlp.loss_curve_), "perdida_final": float(mlp.loss_curve_[-1])}

    cols = ["ubigeo", "departamento", "provincia", "distrito", "ano", "semana", "casos", "umbral", "en_brote", "casos_sig", "umbral_sig", "y"]
    app = t[cols + ["p", "yhat"]].copy()
    app["regla"] = sistema_experto(te)[1].values
    app.to_csv(salida / "app_panel.csv.gz", index=False)


def severidad(df, F, salida):
    d = df[df.ano >= 2011].copy()
    d["dep"] = d.departamento.astype("category").cat.codes; d["sx"] = (d.sexo == "M").astype(int)
    d["sen"] = np.sin(2 * np.pi * d.semana / 52); d["cos"] = np.cos(2 * np.pi * d.semana / 52)
    X = ["edad_anos", "sx", "dep", "sen", "cos"]
    tr, te = d[d.ano <= FIN_VALID], d[d.ano > FIN_VALID]
    yb = lambda x: (~x.severidad.str.startswith("SIN")).astype(int)
    m = HGB(max_iter=150, class_weight="balanced", categorical_features=[2], random_state=SEED).fit(tr[X], yb(tr))
    p = m.predict_proba(te[X])[:, 1]
    R["severidad_binaria"] = {"pr_auc": float(average_precision_score(yb(te), p)), "roc_auc": float(roc_auc_score(yb(te), p)), "prevalencia": float(yb(te).mean()), "n_test": int(len(te)), "n_train": int(len(tr))}
    m3 = HGB(max_iter=150, class_weight="balanced", categorical_features=[2], random_state=SEED).fit(tr[X], tr.severidad)
    y3 = m3.predict(te[X])
    R["severidad_3clases"] = {"f1_macro": float(f1_score(te.severidad, y3, average="macro")), "recall_grave": float(recall_score(te.severidad == "GRAVE", y3 == "GRAVE")), "precision_grave": float(precision_score(te.severidad == "GRAVE", y3 == "GRAVE", zero_division=0)), "bal_acc": float(balanced_accuracy_score(te.severidad, y3))}


def no_supervisado(df, p, F, salida):
    q = p[p.ano <= FIN_VALID]
    a = q.groupby("ubigeo").agg(casos_sem=("casos", "mean"), pct_semanas=("casos", lambda s: (s > 0).mean()), de=("casos", "std"), brotes=("en_brote", "sum"), departamento=("departamento", "first"), distrito=("distrito", "first"))
    a["cv"] = a.de / a.casos_sem.replace(0, np.nan)
    sem = q.assign(w=2 * np.pi * q.semana / 52, sn=lambda x: np.sin(x.w) * x.casos, cs=lambda x: np.cos(x.w) * x.casos).groupby("ubigeo")[["sn", "cs", "casos"]].sum()
    a["est_sen"] = sem.sn / sem.casos; a["est_cos"] = sem.cs / sem.casos
    a["edad_media"] = df[df.ano <= FIN_VALID].groupby("ubigeo").edad_anos.mean()
    a = a.dropna()
    cols = ["casos_sem", "pct_semanas", "cv", "brotes", "est_sen", "est_cos", "edad_media"]
    X = a[cols].copy(); X[["casos_sem", "brotes"]] = np.log1p(X[["casos_sem", "brotes"]])
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=SEED); xy = pca.fit_transform(Z)
    sil = {k: float(silhouette_score(Z, KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Z))) for k in range(2, 9)}
    k = max(sil, key=sil.get)
    a["cluster"] = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Z)
    mejor = None
    for eps in np.arange(.5, 2.01, .1):
        l = DBSCAN(eps=eps, min_samples=8).fit_predict(Z); nc = len(set(l)) - (-1 in l)
        if nc >= 2:
            s = silhouette_score(Z[l != -1], l[l != -1])
            if mejor is None or s > mejor["sil"]:
                mejor = {"eps": float(eps), "sil": float(s), "clusters": int(nc), "ruido": float((l == -1).mean())}
    R["no_supervisado"] = {"n_distritos": int(len(a)), "silhouette_por_k": sil, "k_elegido": int(k), "pca_varianza": [float(v) for v in pca.explained_variance_ratio_], "dbscan": mejor,
                           "perfil_clusters": a.groupby("cluster")[cols].mean().assign(n=a.groupby("cluster").size()).reset_index().to_dict(orient="records"),
                           "ejemplos": {int(c): a[a.cluster == c].sort_values("casos_sem", ascending=False).head(4).distrito.tolist() for c in sorted(a.cluster.unique())}}
    a.assign(pc1=xy[:, 0], pc2=xy[:, 1]).reset_index().to_csv(salida / "clusters_distritos.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.6, 3.5)); ax.scatter(xy[:, 0], xy[:, 1], c=a.cluster, cmap="tab10", s=12); ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", title=f"Perfiles de distritos: K-Means k={k} sobre PCA"); guardar(fig, F, "14_clusters")
    fig, ax = plt.subplots(figsize=(4.6, 2.6)); ax.plot(list(sil), list(sil.values()), "o-", color="#3a6ea5"); ax.set(xlabel="k", ylabel="Silhouette", title="Selección de k"); guardar(fig, F, "15_silhouette")


def main():
    if len(sys.argv) != 3 or not Path(sys.argv[1]).is_file():
        sys.exit("Uso: pipeline.py <dengue.csv> <dir_salida>")
    salida = Path(sys.argv[2]); F = salida / "fig"; F.mkdir(parents=True, exist_ok=True)
    log("cargando datos"); crudo, df = cargar(sys.argv[1])
    crudo.head(8).to_csv(salida / "evidencia_dataset_original.csv", index=False)
    log("panel distrito-semana"); p, d = construir_panel(df)
    d[["ubigeo", "departamento", "ano", "semana", "casos", "umbral", "en_brote"] + FEATS + ["y"]].tail(8).to_csv(salida / "evidencia_dataset_procesado.csv", index=False)
    log("eda y senales"); eda(df, p, F); senales(df, F)
    log("no supervisado"); no_supervisado(df, p, F, salida)
    log("severidad"); severidad(df, F, salida)
    modelos_supervisados(d, F, salida)
    (salida / "metricas.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    log("LISTO")


if __name__ == "__main__":
    main()
