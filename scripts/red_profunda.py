#!/usr/bin/env python3
"""Redes en PyTorch (GPU MPS): CNN sobre imagen historica del distrito y GRU sobre secuencia de 52 semanas.

Uso: red_profunda.py <dengue.csv> <dir_clima> <dir_v2> [epocas]
Requiere haber corrido experimentos_v2.py. Seleccion y umbrales en validacion; la prueba solo se reporta.
Imagen por fila: 6 anos (filas) x 32 semanas (columnas) x 2 canales (casos, lluvia).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experimentos_v2 as ex  # noqa: E402
import pipeline as pl  # noqa: E402
from sklearn.metrics import average_precision_score as ap  # noqa: E402

torch.manual_seed(pl.SEED)
np.random.seed(pl.SEED)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TAB = pl.FEATS + ex.CLIMA
ANCHO, ANOS = 32, 6


def series(p, c, dir_clima):
    """Devuelve dict ubigeo -> matriz (T, 4) con log casos, lluvia, temperatura y humedad; T = semanas desde ano0."""
    ano0 = int(p.ano.min())
    m = pd.read_csv(Path(dir_clima) / "celda_por_ubigeo.csv", dtype={"ubigeo": str}).set_index("ubigeo")
    cl = {k: v for k, v in c.groupby(["lat_c", "lon_c"])}
    out = {}
    for u, g in p.groupby("ubigeo"):
        g = g.sort_values(["ano", "semana"])
        x = g[["ano", "semana"]].merge(cl[(m.loc[u, "lat_c"], m.loc[u, "lon_c"])][["ano", "semana", "ll", "t", "rh"]], on=["ano", "semana"], how="left")
        out[u] = np.column_stack([np.log1p(g.casos.values), np.log1p(x.ll.fillna(0).values), x.t.ffill().bfill().values, x.rh.ffill().bfill().values]).astype(np.float32)
    return out, ano0


def construir(d, S, ano0):
    n = len(d)
    im = np.zeros((n, 2, ANOS, ANCHO), np.float32)
    sq = np.zeros((n, 52, 4), np.float32)
    idx = ((d.ano.values - ano0) * 52 + d.semana.values - 1).astype(int)
    for i, (u, t) in enumerate(zip(d.ubigeo.values, idx)):
        s = S[u]
        for k in range(ANOS):
            fin = t - 52 * k + (6 if k else 0)  # los anos previos ven 6 semanas por delante: ya ocurrieron
            ini = fin - (ANCHO - 1 if k else 25)
            a, b = max(ini, 0), min(fin + 1, len(s))
            if b > a:
                im[i, :, k, a - ini:b - ini] = s[a:b, :2].T
        a = max(t - 51, 0)
        sq[i, 52 - (t + 1 - a):] = s[a:t + 1]
    return im, sq


def tensores(d, S, ano0):
    """Tensores [imagen, secuencia, tabular] escalados solo con entrenamiento e indices de cada particion."""
    im, sq = construir(d, S, ano0)
    itr = np.where(d.ano <= pl.FIN_TRAIN)[0]; iva = np.where((d.ano > pl.FIN_TRAIN) & (d.ano <= pl.FIN_VALID))[0]; ite = np.where(d.ano > pl.FIN_VALID)[0]
    im[:, 1] = (im[:, 1] - im[itr, 1].mean()) / im[itr, 1].std()
    me, de = sq[itr].mean((0, 1)), sq[itr].std((0, 1))
    sq = (sq - me) / de
    tab = d[TAB].values.astype(np.float32); mt, dt_ = tab[itr].mean(0), tab[itr].std(0)
    tab = (tab - mt) / np.where(dt_ == 0, 1, dt_)
    return [im, sq, tab], itr, iva, ite


class CNN(nn.Module):
    def __init__(self, ntab):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2)),
                               nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.f = nn.Sequential(nn.Linear(64 + ntab, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 1))

    def forward(self, im, sq, tab):
        return self.f(torch.cat([self.c(im), tab], 1)).squeeze(1)


class GRU(nn.Module):
    def __init__(self, ntab):
        super().__init__()
        self.g = nn.GRU(4, 64, num_layers=2, batch_first=True, dropout=.2)
        self.f = nn.Sequential(nn.Linear(64 + ntab, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 1))

    def forward(self, im, sq, tab):
        _, h = self.g(sq)
        return self.f(torch.cat([h[-1], tab], 1)).squeeze(1)


def predecir(m, D, idx, b=8192):
    m.eval(); s = []
    with torch.no_grad():
        for i in range(0, len(idx), b):
            j = idx[i:i + b]
            s.append(torch.sigmoid(m(*[torch.from_numpy(a[j]).to(DEV) for a in D])).cpu().numpy())
    return np.concatenate(s)


def entrenar(cls, D, y, itr, iva, epocas, nombre):
    m = cls(D[2].shape[1]).to(DEV)
    op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(op, 3e-3, total_steps=epocas * (len(itr) // 2048 + 1))
    bce = nn.BCEWithLogitsLoss()
    hist, mejor, est, pac = [], -1, None, 0
    rng = np.random.RandomState(pl.SEED)
    for e in range(epocas):
        t0 = time.time(); m.train(); per = rng.permutation(itr); tot = 0.0
        for i in range(0, len(per), 2048):
            j = np.sort(per[i:i + 2048])
            yb = torch.from_numpy(y[j]).to(DEV)
            l = bce(m(*[torch.from_numpy(a[j]).to(DEV) for a in D]), yb)
            op.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 2.0); op.step(); sch.step()
            tot += l.item() * len(j)
        pv = ap(y[iva], predecir(m, D, iva))
        hist.append({"epoca": e + 1, "perdida": tot / len(per), "pr_auc_valid": float(pv), "seg": time.time() - t0})
        print(nombre, hist[-1], flush=True)
        if pv > mejor:
            mejor, pac = pv, 0
            est = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            pac += 1
            if pac >= 5:
                break
    m.load_state_dict(est)
    return m, hist


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    dengue, dc, v2 = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    epocas = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    _, df = pl.cargar(dengue)
    p, _ = pl.construir_panel(df)
    c = ex.clima_semanal(dc, pl.FIN_TRAIN)
    S, ano0 = series(p, c, dc)
    d = pd.read_pickle(v2 / "panel_v2.pkl").reset_index(drop=True)
    print("construyendo tensores", len(d), DEV, flush=True)
    D, itr, iva, ite = tensores(d, S, ano0)
    y = d.y.values.astype(np.float32)
    sc = np.load(v2 / "scores_tabulares.npz")
    base = max(["una_etapa_sin_clima", "una_etapa_con_clima"], key=lambda k: ap(y[iva], sc[f"{k}_valid"]))
    print("base tabular elegida en validacion:", base, flush=True)
    sg_v, sg_t = sc[f"{base}_valid"], sc[f"{base}_test"]
    va, te = d.iloc[iva], d.iloc[ite]
    res, curvas, sals = [], {}, {"Gradient Boosting (tabular)": (sg_v, sg_t)}
    for nom, cls in [("CNN imagen historica", CNN), ("GRU secuencia", GRU)]:
        t0 = time.time()
        m, h = entrenar(cls, D, y, itr, iva, epocas, nom)
        curvas[nom] = h
        sals[nom] = (predecir(m, D, iva), predecir(m, D, ite))
        torch.save(m.state_dict(), v2 / f"{nom.split()[0].lower()}.pt")
        print(nom, "total", round(time.time() - t0), "s", flush=True)
    for nom in ["CNN imagen historica", "GRU secuencia"]:
        sals[f"Ensamble GB + {nom.split()[0]}"] = ((sg_v + sals[nom][0]) / 2, (sg_t + sals[nom][1]) / 2)
    for nom, (sv, st) in sals.items():
        th = pl.corte_optimo(va.y, sv)
        r = {"modelo": nom, "umbral": th}
        r.update({f"valid_{k}": v for k, v in ex.metricas(va, (sv >= th).astype(int), sv).items()})
        r.update({f"test_{k}": v for k, v in ex.metricas(te, (st >= th).astype(int), st).items()})
        res.append(r)
    t = pd.DataFrame(res); t.to_csv(v2 / "profundas.csv", index=False)
    (v2 / "curvas_redes.json").write_text(json.dumps(curvas, indent=1), encoding="utf-8")
    np.savez(v2 / "scores_redes.npz", **{k.replace(" ", "_") + "_valid": v[0] for k, v in sals.items()}, **{k.replace(" ", "_") + "_test": v[1] for k, v in sals.items()})
    print(t[["modelo", "valid_pr_auc", "valid_f1", "valid_pr_auc_inicio", "test_pr_auc", "test_f1", "test_pr_auc_inicio", "test_recall_inicio", "test_precision_inicio"]].round(3).to_string())


if __name__ == "__main__":
    main()
