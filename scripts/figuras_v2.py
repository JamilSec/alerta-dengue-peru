#!/usr/bin/env python3
"""Figuras de la fase v2. Uso: figuras_v2.py <dengue.csv> <dir_clima> <dir_v2> <dir_fig>"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experimentos_v2 as ex  # noqa: E402
import pipeline as pl  # noqa: E402
import red_profunda as rp  # noqa: E402

AZUL, ROJO, GRIS = "#3a6ea5", "#b5413a", "#9a9a9a"


def guardar(fig, F, n):
    fig.savefig(F / f"{n}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def imagen(dengue, dc, v2, F):
    _, df = pl.cargar(dengue)
    p, _ = pl.construir_panel(df)
    c = ex.clima_semanal(dc, pl.FIN_TRAIN)
    d = pd.read_pickle(v2 / "panel_v2.pkl").reset_index(drop=True)
    f = d[(d.ano > pl.FIN_VALID) & (d.en_brote == 0) & (d.y == 1)].sort_values("casos_sig", ascending=False).head(1)
    S, ano0 = rp.series(p[p.ubigeo == f.ubigeo.iloc[0]], c, dc)
    im, _ = rp.construir(f, S, ano0)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.6), gridspec_kw=dict(wspace=.42))
    for a, m, t, cmap in zip(ax, im[0], ["log(1 + casos)", "log(1 + lluvia semanal, mm)"], ["Reds", "Blues"]):
        h = a.imshow(m, aspect="auto", cmap=cmap)
        a.set(xlabel="Semana relativa (ventana de 32)", title=t, yticks=range(rp.ANOS), yticklabels=["Actual"] + [f"−{k} año" + ("s" if k > 1 else "") for k in range(1, rp.ANOS)])
        fig.colorbar(h, ax=a, fraction=.04)
    fig.suptitle(f"Imagen de entrada a la CNN: {f.distrito.iloc[0]}, {f.ano.iloc[0]}-S{f.semana.iloc[0]:02d}, justo antes de un brote", fontsize=10, y=1.06)
    guardar(fig, F, "16_imagen_cnn")


def curvas(v2, F):
    c = json.loads((v2 / "curvas_redes.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for (n, h), col in zip(c.items(), [AZUL, ROJO]):
        x = [e["epoca"] for e in h]; y = [e["pr_auc_valid"] for e in h]
        ax.plot(x, y, color=col, lw=2, label=n)
    ax.set(xlabel="Época", ylabel="PR-AUC en validación", title="Entrenamiento de las redes (GPU Apple M5)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    guardar(fig, F, "17_curvas_redes")


def bosque(v2, F):
    t = pd.read_csv(v2 / "comparacion_bootstrap.csv").sort_values("test_pr_auc")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(t))
    ax.errorbar(t.test_pr_auc, y, xerr=[t.test_pr_auc - t.ic95_inf, t.ic95_sup - t.test_pr_auc], fmt="o", color=AZUL, ms=7, lw=2, capsize=0)
    ax.set_yticks(y, t.modelo)
    ax.set(xlabel="PR-AUC en prueba (IC 95 %, bootstrap por distrito)", title="Comparación de modelos con incertidumbre")
    ax.spines[["top", "right"]].set_visible(False)
    guardar(fig, F, "18_bootstrap")


def agente(v2, F):
    a = json.loads((v2 / "agente.json").read_text(encoding="utf-8"))
    t = pd.DataFrame(a["politicas"]).sort_values("cobertura_brotes")
    col = [ROJO if p.startswith("Agente") else (GRIS if p.startswith(("Oráculo", "Aleatorio")) else AZUL) for p in t.politica]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.7), sharey=True)
    for a_, m, ti in zip(ax, ["cobertura_brotes", "cobertura_inicios"], ["Brotes atendidos a tiempo", "Inicios de brote atendidos"]):
        a_.barh(t.politica, t[m] * 100, color=col, height=.62)
        for i, v in enumerate(t[m] * 100):
            a_.text(v + 1, i, f"{v:.1f} %".replace(".", ","), va="center", fontsize=8, color="#333")
        a_.set(title=ti, xlabel="% del total en prueba", xlim=(0, 100))
        a_.spines[["top", "right"]].set_visible(False)
    guardar(fig, F, "19_agente_cobertura")
    p = pd.read_csv(v2 / "agente_plan.csv.gz", dtype={"ubigeo": str})
    sem = p.groupby(["ano", "semana"]).y.sum().idxmax()
    w = p[(p.ano == sem[0]) & (p.semana == sem[1])]
    u = pd.read_csv(sys.argv[2] + "/celda_por_ubigeo.csv", dtype={"ubigeo": str})
    d = pd.read_pickle(v2 / "panel_v2.pkl")[["ubigeo"]].drop_duplicates().merge(u, on="ubigeo")
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.scatter(d.longitude, d.latitude, s=4, color="#d4d4d4", label="Distritos vigilados")
    for _, g in w.groupby("departamento"):
        g = g.sort_values("orden")
        ax.plot([g.lon_base.iloc[0], *g.lon, g.lon_base.iloc[0]], [g.lat_base.iloc[0], *g.lat, g.lat_base.iloc[0]], color=GRIS, lw=1, zorder=2)
    ax.scatter(w.lon_base, w.lat_base, marker="s", s=26, color="#333", label="Base de brigada", zorder=4)
    ve = w[w.y == 1]; fa = w[w.y == 0]
    ax.scatter(ve.lon, ve.lat, s=34, color=ROJO, edgecolor="white", lw=.6, label="Visita: hubo brote", zorder=5)
    ax.scatter(fa.lon, fa.lat, s=34, color=AZUL, edgecolor="white", lw=.6, label="Visita: sin brote", zorder=5)
    ax.set(xlabel="Longitud", ylabel="Latitud", title=f"Plan del agente: {sem[0]}, semana {sem[1]:02d}", aspect="equal")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    guardar(fig, F, "20_agente_plan")


def limpio(ax):
    ax.spines[["top", "right"]].set_visible(False)


def barras(ax, x, series, titulo, ylim=(0, 1)):
    n, w = len(series), .8 / len(series)
    for k, (nom, val, col) in enumerate(series):
        pos = np.arange(len(x)) + (k - (n - 1) / 2) * w
        ax.bar(pos, val, w * .9, color=col, label=nom)
        for a, b in zip(pos, val):
            ax.text(a, b + .015, f"{b:.2f}".replace(".", ","), ha="center", fontsize=7, color="#333")
    ax.set_xticks(np.arange(len(x)), x); ax.set(title=titulo, ylim=ylim); ax.legend(frameon=False, fontsize=7, loc="upper center", ncols=n, bbox_to_anchor=(.5, -.1)); limpio(ax)


def gru(v2, F):
    A = json.loads((v2 / "analisis_gru.json").read_text(encoding="utf-8"))
    S = json.loads((v2 / "sistema_final.json").read_text(encoding="utf-8"))
    m = np.array(S["matriz_confusion"])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.25)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:,}".replace(",", " "), ha="center", va="center", color="white" if v > m.max() * .6 else "#222")
    ax.set_xticks([0, 1], ["No brote", "Brote"]); ax.set_yticks([0, 1], ["No brote", "Brote"]); ax.set(xlabel="Predicho", ylabel="Real", title="Matriz de confusión de la GRU (prueba)")
    ax.grid(False); guardar(fig, F, "21_matriz_gru")

    t = pd.DataFrame(A["importancia"]).sort_values("caida_pr_auc")
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3), sharey=True)
    for a, c, ti in zip(ax, ["caida_pr_auc", "caida_pr_auc_inicio"], ["Todas las semanas", "Solo semanas sin brote (inicios)"]):
        a.barh(t.grupo, t[c], color=AZUL, height=.62); a.set(title=ti, xlabel="Caída de PR-AUC al permutar"); limpio(a)
    guardar(fig, F, "22_importancia_gru")

    p = pd.read_csv(v2 / "app_panel_v2.csv.gz", dtype={"ubigeo": str}, keep_default_na=False, na_values=[""])
    u = p.groupby("ubigeo").y.sum().idxmax()
    g = p[p.ubigeo == u].sort_values(["ano", "semana"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    ax.plot(g.index, g.casos_sig, color="#333", label="Casos (semana siguiente)"); ax.plot(g.index, g.umbral_sig, color="#e0a030", ls="--", label="Umbral")
    al = g[g.yhat == 1]; ax.scatter(al.index, al.casos_sig, color=ROJO, s=12, label="Alerta de la GRU", zorder=3)
    ax.set(title=f"Ejemplo: {g.distrito.iloc[0]} ({g.departamento.iloc[0]}), 2023-2025", xlabel="Semana del periodo", ylabel="Casos"); ax.legend(frameon=False, fontsize=7); limpio(ax)
    guardar(fig, F, "23_ejemplo_gru")

    a = pd.DataFrame(A["por_ano"])
    x = a.ano.astype(str).tolist()
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.9))
    barras(ax[0], x, [("GRU", a.pr_auc, AZUL), ("Azar", a.prevalencia, GRIS)], "PR-AUC")
    barras(ax[1], x, [("GRU", a.f1, AZUL), ("Persistencia", a.f1_persistencia, GRIS)], "F1")
    barras(ax[2], x, [("Recall", a.recall_inicio, AZUL), ("Precisión", a.precision_inicio, ROJO)], "Inicios de brote", (0, .7))
    fig.tight_layout(); guardar(fig, F, "24_por_ano")

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 3.6), sharex=True)
    s = p.sort_values(["ano", "semana"])
    cl = s.groupby(["ano", "semana"], sort=True)
    nu = s[(s.en_brote == 0) & (s.yhat == 1)].groupby(["ano", "semana"]).size().reindex(cl.size().index, fill_value=0)
    co = s[(s.en_brote == 1) & (s.yhat == 1)].groupby(["ano", "semana"]).size().reindex(cl.size().index, fill_value=0)
    t_ = np.arange(len(nu))
    ax[0].fill_between(t_, nu.values, color=ROJO, step="mid", lw=0); ax[0].set(ylabel="Alertas", title="Alertas de inicio por semana (distritos sin brote)")
    ax[1].fill_between(t_, co.values, color=AZUL, step="mid", lw=0); ax[1].set(ylabel="Alertas", title="Alertas de continuación por semana (distritos en brote)")
    for a_ in ax:
        limpio(a_)
    ax[1].set_xticks(np.arange(0, len(nu), 52), ["2023", "2024", "2025"])
    fig.tight_layout(); guardar(fig, F, "25_carga_alertas")

    h = pd.DataFrame(A["horizontes"])
    col = {"GRU": ROJO, "Gradient Boosting": AZUL, "Persistencia": GRIS}
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.8))
    for a_, c, ti in zip(ax, ["f1", "pr_auc_inicio", "recall_inicio"], ["F1", "PR-AUC en inicios", "Recall de inicios"]):
        for nom, g_ in h.groupby("modelo"):
            g_ = g_.dropna(subset=[c])
            if len(g_) and not (c == "recall_inicio" and nom == "Persistencia"):
                a_.plot(g_.horizonte, g_[c], "o-", color=col[nom], lw=2, ms=6, label=nom)
        a_.set(title=ti, xlabel="Semanas de anticipación", xticks=[1, 2, 4], ylim=(0, 1)); limpio(a_)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); guardar(fig, F, "26_horizontes")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    v2, F = Path(sys.argv[3]), Path(sys.argv[4])
    imagen(sys.argv[1], sys.argv[2], v2, F); curvas(v2, F); bosque(v2, F); agente(v2, F); gru(v2, F)
    print("OK")
