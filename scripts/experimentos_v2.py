#!/usr/bin/env python3
"""Experimentos v2: clima NASA POWER y modelo de dos etapas para inicios de brote.

Uso: experimentos_v2.py <dengue.csv> <dir_clima> <dir_salida>
Todas las decisiones (variables, hiperparametros, umbrales) se toman en validacion.
La prueba solo se reporta. Salida: variantes.csv, panel_v2.parquet o .pkl, resumen en consola.
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier as HGB  # noqa: E402
from sklearn.metrics import average_precision_score as ap, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import ParameterGrid  # noqa: E402

SEED = pl.SEED
CLIMA = ["ll_0", "ll_4", "ll_lag4_8", "ll_12", "ll_12_an", "t_4", "t_8", "t_4_an", "tmax_4", "rh_4", "rh_8", "altitud"]


def inicio_se(y):
    """Domingo en que empieza la semana epidemiologica 1: la SE1 tiene al menos 4 dias en enero."""
    e1 = dt.date(y, 1, 1)
    d = (e1.weekday() + 1) % 7  # domingo = 0
    return e1 - dt.timedelta(days=d) if d <= 3 else e1 + dt.timedelta(days=7 - d)


def semana_epi(f):
    y = f.year + 1 if f >= inicio_se(f.year + 1) else f.year
    if f < inicio_se(y):
        y -= 1
    return y, min((f - inicio_se(y)).days // 7 + 1, 52)


def clima_semanal(dir_clima, fin_clim):
    cal = {}
    filas = []
    for ruta in sorted((Path(dir_clima) / "celdas").glob("*.csv")):
        la, lo = ruta.stem.split("_")
        d = pd.read_csv(ruta, dtype={"fecha": str})
        if not cal:
            for f in d.fecha:
                cal[f] = semana_epi(dt.date(int(f[:4]), int(f[4:6]), int(f[6:])))
        d["ano"] = d.fecha.map(lambda f: cal[f][0]); d["semana"] = d.fecha.map(lambda f: cal[f][1])
        s = d.groupby(["ano", "semana"]).agg(ll=("PRECTOTCORR", "sum"), t=("T2M", "mean"), tmax=("T2M_MAX", "mean"), rh=("RH2M", "mean")).reset_index()
        s["lat_c"], s["lon_c"] = float(la), float(lo)
        filas.append(s)
    c = pd.concat(filas).sort_values(["lat_c", "lon_c", "ano", "semana"]).reset_index(drop=True)
    g = c.groupby(["lat_c", "lon_c"])
    c["ll_0"] = c["ll"]
    c["ll_4"] = g["ll"].transform(lambda s: s.rolling(4).sum())
    c["ll_12"] = g["ll"].transform(lambda s: s.rolling(12).sum())
    c["ll_lag4_8"] = g["ll"].transform(lambda s: s.shift(4).rolling(5).sum())
    c["t_4"] = g["t"].transform(lambda s: s.rolling(4).mean())
    c["t_8"] = g["t"].transform(lambda s: s.rolling(8).mean())
    c["tmax_4"] = g["tmax"].transform(lambda s: s.rolling(4).mean())
    c["rh_4"] = g["rh"].transform(lambda s: s.rolling(4).mean())
    c["rh_8"] = g["rh"].transform(lambda s: s.rolling(8).mean())
    # anomalias contra la climatologia solo de los anos de entrenamiento
    base = c[c.ano <= fin_clim].groupby(["lat_c", "lon_c", "semana"])[["ll_12", "t_4"]].mean().rename(columns={"ll_12": "cl_ll", "t_4": "cl_t"}).reset_index()
    c = c.merge(base, on=["lat_c", "lon_c", "semana"], how="left")
    c["ll_12_an"] = c.ll_12 - c.cl_ll
    c["t_4_an"] = c.t_4 - c.cl_t
    return c


def unir_clima(d, c, dir_clima):
    m = pd.read_csv(Path(dir_clima) / "celda_por_ubigeo.csv", dtype={"ubigeo": str})
    d = d.merge(m[["ubigeo", "lat_c", "lon_c", "altitude"]].rename(columns={"altitude": "altitud"}), on="ubigeo", how="left")
    cols = ["lat_c", "lon_c", "ano", "semana"] + [x for x in CLIMA if x != "altitud"]
    return d.merge(c[cols], on=["lat_c", "lon_c", "ano", "semana"], how="left")


def metricas(d, yhat, s):
    y = d.y.values
    ini = d.en_brote.values == 0
    return {"pr_auc": ap(y, s), "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
            "pr_auc_inicio": ap(y[ini], s[ini]), "recall_inicio": float(yhat[ini & (y == 1)].mean()), "precision_inicio": precision_score(y[ini], yhat[ini], zero_division=0),
            "alertas_inicio": int(yhat[ini].sum())}


def mejores_params(tr, feats, peso=None):
    """Grid con pliegues temporales de los ultimos 3 anos de entrenamiento."""
    mejor, mp = None, -1
    for g in ParameterGrid({"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0], "min_samples_leaf": [20, 60], "max_iter": [300]}):
        v = []
        for a in range(pl.FIN_TRAIN - 2, pl.FIN_TRAIN + 1):
            x, z = tr[tr.ano < a], tr[tr.ano == a]
            v.append(ap(z.y, HGB(random_state=SEED, class_weight=peso, **g).fit(x[feats], x.y).predict_proba(z[feats])[:, 1]))
        if np.mean(v) > mp:
            mp, mejor = float(np.mean(v)), g
    return mejor, mp


def una_etapa(tr, va, te, feats, pars):
    m = HGB(random_state=SEED, **pars).fit(tr[feats], tr.y)
    sv, st = m.predict_proba(va[feats])[:, 1], m.predict_proba(te[feats])[:, 1]
    th = pl.corte_optimo(va.y, sv)
    return sv, st, (sv >= th).astype(int), (st >= th).astype(int), {"umbral": th}


def dos_etapas(tr, va, te, feats, p0, p1):
    """Etapa 0: distritos sin brote (inicio). Etapa 1: distritos en brote (continuacion). Umbral propio por etapa."""
    f0, f1_ = tr[tr.en_brote == 0], tr[tr.en_brote == 1]
    m0 = HGB(random_state=SEED, **p0).fit(f0[feats], f0.y)
    m1 = HGB(random_state=SEED, **p1).fit(f1_[feats], f1_.y)

    def puntuar(x):
        s = np.zeros(len(x))
        a = x.en_brote.values == 0
        s[a] = m0.predict_proba(x.loc[a, feats])[:, 1]
        s[~a] = m1.predict_proba(x.loc[~a, feats])[:, 1]
        return s, a

    sv, av = puntuar(va); st, at = puntuar(te)
    th0, th1 = pl.corte_optimo(va.y[av], sv[av]), pl.corte_optimo(va.y[~av], sv[~av])
    h = lambda s, a: ((a & (s >= th0)) | (~a & (s >= th1))).astype(int)
    return sv, st, h(sv, av), h(st, at), {"umbral_inicio": th0, "umbral_continuacion": th1}


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sal = Path(sys.argv[3]); sal.mkdir(parents=True, exist_ok=True)
    _, df = pl.cargar(sys.argv[1])
    _, d = pl.construir_panel(df)
    t0 = time.time()
    c = clima_semanal(sys.argv[2], pl.FIN_TRAIN)
    d = unir_clima(d, c, sys.argv[2])
    nul = d[CLIMA].isna().mean().max()
    print(f"panel {len(d)} nulos clima max {nul:.4f} clima {time.time() - t0:.0f}s", flush=True)
    d = d.dropna(subset=CLIMA).copy()
    tr, va, te = d[d.ano <= pl.FIN_TRAIN], d[(d.ano > pl.FIN_TRAIN) & (d.ano <= pl.FIN_VALID)], d[d.ano > pl.FIN_VALID]
    con = pl.FEATS + CLIMA
    salida, res = {}, []
    params = {}
    for nom, fs in [("sin_clima", pl.FEATS), ("con_clima", con)]:
        t0 = time.time()
        p_u, _ = mejores_params(tr, fs)
        p_0, _ = mejores_params(tr[tr.en_brote == 0], fs)
        p_1, _ = mejores_params(tr[tr.en_brote == 1], fs)
        params[nom] = {"una_etapa": p_u, "inicio": p_0, "continuacion": p_1}
        print(nom, "grid", round(time.time() - t0), "s", flush=True)
        for var, fn in [("una etapa", lambda: una_etapa(tr, va, te, fs, p_u)), ("dos etapas", lambda: dos_etapas(tr, va, te, fs, p_0, p_1))]:
            sv, st, hv, ht, ex = fn()
            r = {"variante": f"{var} {nom}"}
            r.update({f"valid_{k}": v for k, v in metricas(va, hv, sv).items()})
            r.update({f"test_{k}": v for k, v in metricas(te, ht, st).items()})
            r.update(ex)
            res.append(r)
            salida[r["variante"]] = (sv, st)
    t = pd.DataFrame(res)
    t.to_csv(sal / "variantes.csv", index=False)
    (sal / "params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    d.to_pickle(sal / "panel_v2.pkl")
    np.savez(sal / "scores_tabulares.npz", **{k.replace(" ", "_") + "_valid": v[0] for k, v in salida.items()}, **{k.replace(" ", "_") + "_test": v[1] for k, v in salida.items()})
    print(t[["variante", "valid_pr_auc", "valid_f1", "valid_pr_auc_inicio", "valid_recall_inicio", "valid_precision_inicio"]].round(3).to_string())
    print(t[["variante", "test_pr_auc", "test_f1", "test_pr_auc_inicio", "test_recall_inicio", "test_precision_inicio", "test_alertas_inicio"]].round(3).to_string())


if __name__ == "__main__":
    main()
