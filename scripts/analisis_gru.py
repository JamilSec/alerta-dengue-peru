#!/usr/bin/env python3
"""Analisis del sistema final (GRU): entrenamiento, importancia, errores, desempeno por ano, carga de alertas y horizonte.

Uso: analisis_gru.py <dengue.csv> <dir_clima> <dir_v2> [epocas]
Requiere red_profunda.py y agente.py. Umbrales fijados en validacion; la prueba solo se reporta.
Salida: analisis_gru.json y errores_departamento_gru.csv en <dir_v2>.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluar_v2 as ev  # noqa: E402
import experimentos_v2 as ex  # noqa: E402
import pipeline as pl  # noqa: E402
import red_profunda as rp  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier as HGB  # noqa: E402
from sklearn.metrics import average_precision_score as ap, f1_score  # noqa: E402

GRUPOS = {
    "Casos recientes": ["lg_l0", "lg_l1", "lg_l2", "lg_l3", "lg_media4", "lg_max8", "tendencia"],
    "Cercanía al umbral": ["ratio", "en_brote", "brotes4", "lg_umbral_sig", "lg_hist_sig"],
    "Contexto departamental": ["lg_dep_otros", "dep_brotes"],
    "Estacionalidad": ["sen", "cos"],
    "Clima reciente": ex.CLIMA,
}
CANALES = {"Historia de casos (52 semanas)": [0], "Lluvia (52 semanas)": [1], "Temperatura y humedad (52 semanas)": [2, 3]}
PRECISION_INICIO = 0.40


def alertar(d, s, th0, th1):
    a = d.en_brote.values == 0
    return ((a & (s >= th0)) | (~a & (s >= th1))).astype(int)


def importancia(m, im, sq, tab, te, repeticiones=5):
    """Caida de PR-AUC (global y en inicios) al permutar en conjunto cada grupo de entradas."""
    y, ini = te.y.values, te.en_brote.values == 0
    idx = np.arange(len(te))
    base = rp.predecir(m, [im, sq, tab], idx)
    b_g, b_i = ap(y, base), ap(y[ini], base[ini])
    rng = np.random.RandomState(pl.SEED)
    salida = []
    for nombre, cols, canales in [(k, [rp.TAB.index(c) for c in v], []) for k, v in GRUPOS.items()] + [(k, [], v) for k, v in CANALES.items()]:
        g, i = [], []
        for _ in range(repeticiones):
            per, t2, s2 = rng.permutation(len(te)), tab.copy(), sq.copy()
            if cols:
                t2[:, cols] = tab[per][:, cols]
            if canales:
                s2[:, :, canales] = sq[per][:, :, canales]
            s = rp.predecir(m, [im, s2, t2], idx)
            g.append(b_g - ap(y, s)); i.append(b_i - ap(y[ini], s[ini]))
        salida.append({"grupo": nombre, "caida_pr_auc": float(np.mean(g)), "de": float(np.std(g)), "caida_pr_auc_inicio": float(np.mean(i)), "de_inicio": float(np.std(i))})
        print("importancia", salida[-1], flush=True)
    return sorted(salida, key=lambda x: -x["caida_pr_auc"])


def errores(t, v2):
    t = t.assign(vp=(t.y == 1) & (t.yhat == 1), fp=(t.y == 0) & (t.yhat == 1), fn=(t.y == 1) & (t.yhat == 0))
    e = t.groupby("departamento").agg(filas=("y", "size"), brotes=("y", "sum"), VP=("vp", "sum"), FP=("fp", "sum"), FN=("fn", "sum")).reset_index()
    e["recall"] = e.VP / e.brotes.replace(0, np.nan); e["precision"] = e.VP / (e.VP + e.FP).replace(0, np.nan)
    e = e.sort_values("brotes", ascending=False)
    e.to_csv(v2 / "errores_departamento_gru.csv", index=False)
    return {"FN_media_casos_semana_actual": float(t[t.fn].casos.mean()), "VP_media_casos_semana_actual": float(t[t.vp].casos.mean()),
            "FN_que_son_inicio": float((t[t.fn].en_brote == 0).mean()), "FP_que_estaban_en_brote": float((t[t.fp].en_brote == 1).mean()),
            "FP_inicio": int(((t.fp) & (t.en_brote == 0)).sum()), "FP_continuacion": int(((t.fp) & (t.en_brote == 1)).sum())}


def por_ano(t):
    filas = []
    for a, x in t.groupby("ano"):
        v = x[x.en_brote == 0]
        filas.append({"ano": int(a), "filas": int(len(x)), "prevalencia": float(x.y.mean()), "pr_auc": ap(x.y, x.p), "f1": f1_score(x.y, x.yhat), "f1_persistencia": f1_score(x.y, x.en_brote),
                      "inicios": int(v.y.sum()), "recall_inicio": float(v[v.y == 1].yhat.mean()), "precision_inicio": float(v[v.yhat == 1].y.mean()), "azar_inicio": float(v.y.mean())})
    return filas


def carga(t):
    claves = ["ano", "semana"]
    tot = t.groupby(claves).yhat.sum()
    nuevas = t[(t.en_brote == 0) & (t.yhat == 1)].groupby(claves).size().reindex(tot.index, fill_value=0)
    dep = t[(t.en_brote == 0) & (t.yhat == 1)].groupby(claves + ["departamento"]).size()
    r = lambda s: {"mediana": float(s.median()), "p90": float(s.quantile(.9)), "maximo": int(s.max())}
    v = t[t.en_brote == 0]
    top = v.sort_values("p", ascending=False).groupby(claves).head(10)
    return {"alertas_semana": r(tot), "nuevas_semana": r(nuevas), "semanas": int(len(tot)), "semanas_sin_nuevas": int((nuevas == 0).sum()), "nuevas_por_departamento_activo": r(dep),
            "top10_precision": float(top.y.mean()), "top10_inicios_cubiertos": float(top.y.sum() / v.y.sum()), "azar_inicio": float(v.y.mean())}


def tamano_inicios(t, ventana=8):
    """Casos notificados en las 8 semanas siguientes a cada inicio, segun si la alerta lo detecto."""
    t = t.sort_values(["ubigeo", "ano", "semana"]).copy()
    t["futuro"] = t.groupby("ubigeo").casos.transform(lambda s: s[::-1].rolling(ventana).sum()[::-1].shift(-1))
    i = t[(t.en_brote == 0) & (t.y == 1) & t.futuro.notna()]
    d, f = i[i.yhat == 1].futuro, i[i.yhat == 0].futuro
    return {"ventana_semanas": ventana, "n_detectados": int(len(d)), "n_omitidos": int(len(f)), "mediana_detectados": float(d.median()), "mediana_omitidos": float(f.median()), "media_detectados": float(d.mean()), "media_omitidos": float(f.mean()), "proporcion_casos_en_detectados": float(d.sum() / i.futuro.sum())}


def panel_horizonte(d, p, h):
    """Mismo panel con objetivo en t+h; umbral, media historica y estacionalidad pasan a t+h (se conocen en t)."""
    q = p.sort_values(["ubigeo", "ano", "semana"])
    g = q.groupby("ubigeo")
    fut = q[["ubigeo", "ano", "semana"]].assign(y_h=g.en_brote.shift(-h), um_h=g.umbral.shift(-h), hi_h=g.hist_media.shift(-h), se_h=g.semana.shift(-h))
    x = d.merge(fut, on=["ubigeo", "ano", "semana"], how="left").dropna(subset=["y_h", "um_h", "hi_h", "se_h"]).reset_index(drop=True)
    x["y"] = x.y_h.astype(int); x["lg_umbral_sig"] = np.log1p(x.um_h); x["lg_hist_sig"] = np.log1p(x.hi_h)
    x["sen"] = np.sin(2 * np.pi * x.se_h / 52); x["cos"] = np.cos(2 * np.pi * x.se_h / 52)
    return x.drop(columns=["y_h", "um_h", "hi_h", "se_h"])


def evaluar_h(nombre, h, va, te, sv, st):
    bv, y = va.en_brote.values == 1, va.y.values
    th0, th1 = ev.corte_precision(y[~bv], sv[~bv], PRECISION_INICIO), pl.corte_optimo(y[bv], sv[bv])
    m = ex.metricas(te, alertar(te, st, th0, th1), st)
    return {"modelo": nombre, "horizonte": h, "n_test": int(len(te)), "prevalencia": float(te.y.mean()), "n_inicios": int(te[te.en_brote == 0].y.sum()), "azar_inicio": float(te[te.en_brote == 0].y.mean()), **{k: float(v) for k, v in m.items()}}


def persistencia_h(h, te):
    y, a = te.y.values, te.en_brote.values
    tp = ((y == 1) & (a == 1)).sum()
    return {"modelo": "Persistencia", "horizonte": h, "n_test": int(len(te)), "prevalencia": float(y.mean()), "n_inicios": int(te[te.en_brote == 0].y.sum()), "azar_inicio": float(te[te.en_brote == 0].y.mean()),
            "pr_auc": None, "precision": float(tp / a.sum()), "recall": float(tp / y.sum()), "f1": float(f1_score(y, a)), "pr_auc_inicio": None, "recall_inicio": 0.0, "precision_inicio": 0.0, "alertas_inicio": 0}


def horizontes(d, p, S, ano0, v2, epocas):
    pars = json.loads((v2 / "params.json").read_text(encoding="utf-8"))["sin_clima"]["una_etapa"]
    z = np.load(v2 / "scores_redes.npz")
    filas = []
    for h in [1, 2, 4]:
        x = d if h == 1 else panel_horizonte(d, p, h)
        va, te = x[(x.ano > pl.FIN_TRAIN) & (x.ano <= pl.FIN_VALID)], x[x.ano > pl.FIN_VALID]
        filas.append(persistencia_h(h, te))
        if h == 1:
            sal = {"Gradient Boosting": (z["Gradient_Boosting_(tabular)_valid"], z["Gradient_Boosting_(tabular)_test"]), "GRU": (z["GRU_secuencia_valid"], z["GRU_secuencia_test"])}
        else:
            tr = x[x.ano <= pl.FIN_TRAIN]
            gb = HGB(random_state=pl.SEED, **pars).fit(tr[pl.FEATS], tr.y)
            sal = {"Gradient Boosting": (gb.predict_proba(va[pl.FEATS])[:, 1], gb.predict_proba(te[pl.FEATS])[:, 1])}
            D, itr, iva, ite = rp.tensores(x, S, ano0)
            torch.manual_seed(pl.SEED)
            m, _ = rp.entrenar(rp.GRU, D, x.y.values.astype(np.float32), itr, iva, epocas, f"GRU h={h}")
            sal["GRU"] = (rp.predecir(m, D, iva), rp.predecir(m, D, ite))
            del D
        for nom, (sv, st) in sal.items():
            filas.append(evaluar_h(nom, h, va, te, sv, st))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in filas[-1].items()}, flush=True)
    return filas


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    dengue, dc, v2 = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    epocas = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    F = json.loads((v2 / "sistema_final.json").read_text(encoding="utf-8"))
    if not F["modelo"].startswith("GRU"):
        sys.exit(f"Este analisis asume que el sistema final es la GRU y es: {F['modelo']}")
    _, df = pl.cargar(dengue)
    p, _ = pl.construir_panel(df)
    S, ano0 = rp.series(p, ex.clima_semanal(dc, pl.FIN_TRAIN), dc)
    d = pd.read_pickle(v2 / "panel_v2.pkl").reset_index(drop=True)
    D, itr, iva, ite = rp.tensores(d, S, ano0)
    m = rp.GRU(D[2].shape[1]).to(rp.DEV)
    m.load_state_dict(torch.load(v2 / "gru.pt", map_location=rp.DEV))
    y = d.y.values
    st = rp.predecir(m, D, ite)
    if abs(ap(y[ite], st) - F["pr_auc"]) > 1e-3:
        sys.exit(f"gru.pt no reproduce el sistema final: {ap(y[ite], st):.4f} frente a {F['pr_auc']:.4f}")
    print("reproduce PR-AUC", round(ap(y[ite], st), 4), flush=True)
    tr, s_tr = d.iloc[itr], rp.predecir(m, D, itr)
    h_tr = alertar(tr, s_tr, F["umbral_inicio"], F["umbral_continuacion"])
    entren = {k: v for k, v in pl.met_bin(tr, h_tr, s_tr).items()}
    entren["precision_inicio"] = float(tr.y.values[(tr.en_brote.values == 0) & (h_tr == 1)].mean())
    te = d.iloc[ite]
    imp = importancia(m, D[0][ite], D[1][ite], D[2][ite], te)
    del D, m
    t = pd.read_csv(v2 / "app_panel_v2.csv.gz", dtype={"ubigeo": str}, keep_default_na=False, na_values=[""])
    r = {"entrenamiento": entren, "importancia": imp, "errores": errores(t, v2), "por_ano": por_ano(t), "carga": carga(t), "tamano_inicios": tamano_inicios(t)}
    (v2 / "analisis_gru.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k != "importancia"}, indent=0, ensure_ascii=False), flush=True)
    r["horizontes"] = horizontes(d, p, S, ano0, v2, epocas)
    (v2 / "analisis_gru.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print("LISTO", flush=True)


if __name__ == "__main__":
    main()
