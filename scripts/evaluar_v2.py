#!/usr/bin/env python3
"""Compara modelos v2 con intervalos bootstrap por distrito y fija la alerta de inicio con una precision objetivo.

Uso: evaluar_v2.py <dir_v2> [precision_objetivo] [repeticiones]
Requiere panel_v2.pkl y scores_redes.npz. El umbral se elige en validacion; la prueba solo se reporta.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402
from sklearn.metrics import average_precision_score as ap, precision_recall_curve  # noqa: E402


def corte_precision(y, s, objetivo):
    """Umbral de mayor recall con precision >= objetivo; si no se alcanza, el de maxima precision."""
    pr, rc, th = precision_recall_curve(y, s)
    ok = np.where(pr[:-1] >= objetivo)[0]
    return float(th[ok[np.argmax(rc[:-1][ok])]] if len(ok) else th[int(np.argmax(pr[:-1]))])


def operar(y, s, th):
    h = s >= th
    return {"alertas": int(h.sum()), "precision": float(y[h].mean()) if h.any() else None, "recall": float(h[y == 1].mean())}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    v2 = Path(sys.argv[1])
    obj = float(sys.argv[2]) if len(sys.argv) > 2 else 0.40
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    d = pd.read_pickle(v2 / "panel_v2.pkl").reset_index(drop=True)
    z = np.load(v2 / "scores_redes.npz")
    nombres = sorted({k[:-6] for k in z.files if k.endswith("_valid")})
    va = d[(d.ano > pl.FIN_TRAIN) & (d.ano <= pl.FIN_VALID)]; te = d[d.ano > pl.FIN_VALID]
    grupos = te.ubigeo.values
    u = np.unique(grupos)
    pos = {x: np.where(grupos == x)[0] for x in u}
    yt, it = te.y.values, te.en_brote.values == 0
    rng = np.random.RandomState(42)
    muestras = [np.concatenate([pos[x] for x in rng.choice(u, len(u))]) for _ in range(reps)]
    ref = "Gradient_Boosting_(tabular)"
    filas, oper = [], {}
    for n in nombres:
        s = z[f"{n}_test"]
        b = np.array([ap(yt[m], s[m]) for m in muestras])
        dif = np.array([ap(yt[m], s[m]) - ap(yt[m], z[f"{ref}_test"][m]) for m in muestras]) if n != ref else np.zeros(reps)
        bi = np.array([ap(yt[m][it[m]], s[m][it[m]]) for m in muestras])
        filas.append({"modelo": n.replace("_", " "), "valid_pr_auc": ap(va.y, z[f"{n}_valid"]), "test_pr_auc": ap(yt, s), "ic95_inf": np.percentile(b, 2.5), "ic95_sup": np.percentile(b, 97.5),
                      "dif_vs_gb": float(ap(yt, s) - ap(yt, z[f"{ref}_test"])), "dif_inf": np.percentile(dif, 2.5), "dif_sup": np.percentile(dif, 97.5),
                      "test_pr_auc_inicio": ap(yt[it], s[it]), "inicio_inf": np.percentile(bi, 2.5), "inicio_sup": np.percentile(bi, 97.5)})
        iv = va.en_brote.values == 0
        th = corte_precision(va.y.values[iv], z[f"{n}_valid"][iv], obj)
        oper[n.replace("_", " ")] = {"umbral_inicio": th, "valid": operar(va.y.values[iv], z[f"{n}_valid"][iv], th), "test": operar(yt[it], s[it], th)}
    t = pd.DataFrame(filas).sort_values("valid_pr_auc", ascending=False)
    t.to_csv(v2 / "comparacion_bootstrap.csv", index=False)
    (v2 / "alerta_inicio.json").write_text(json.dumps({"precision_objetivo": obj, "prevalencia_inicio_test": float(yt[it].mean()), "n_inicios_test": int(yt[it].sum()), "modelos": oper}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(t.round(3).to_string())
    print("prevalencia inicio test", round(float(yt[it].mean()), 4), "inicios", int(yt[it].sum()))
    for k, v in oper.items():
        print(k, "| th", round(v["umbral_inicio"], 3), "| valid", v["valid"], "| test", v["test"])


if __name__ == "__main__":
    main()
