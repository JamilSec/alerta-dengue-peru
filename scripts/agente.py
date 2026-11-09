#!/usr/bin/env python3
"""Agente de vigilancia: une riesgo del modelo, reglas expertas y busqueda de rutas para brigadas.

Uso: agente.py <dir_v2> <celda_por_ubigeo.csv> [paradas] [km_max]
Cada semana, por departamento, una brigada sale de la capital departamental y elige hasta <paradas> distritos
con un recorrido de ida y vuelta de hasta <km_max> km en linea recta (problema de orientacion resuelto con A*).
Requiere experimentos_v2.py, red_profunda.py y evaluar_v2.py. Salida: agente.json, agente_plan.csv.gz, sistema_final.json, app_panel_v2.csv.gz
"""
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

N_CAND = 10
TEXTO_REGLA = {n: t for n, _, t in pl.REGLAS}


def km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(h)))


def buscar(costos, D, m, km_max, modo="a*"):
    """Orientacion con costo = suma de (1 - riesgo) y 1 por parada no usada.

    D[0] es la base; los candidatos son 1..n. modo: 'a*', 'ucs' (sin heuristica) o 'voraz' (solo heuristica).
    h suma los (m - paradas) costos mas bajos aun libres: nunca sobreestima, asi que A* es optimo.
    """
    n = len(costos)
    orden = sorted(range(1, n + 1), key=lambda i: costos[i - 1])

    def h(vis, prof):
        falta, tot = m - prof, 0.0
        for i in orden:
            if falta == 0:
                break
            if i not in vis:
                tot += costos[i - 1]; falta -= 1
        return tot + falta

    prio = lambda g, hv: {"a*": g + hv, "ucs": g, "voraz": hv}[modo]
    cola, tie, expandidos = [(prio(0.0, h((), 0)), 0, 0.0, 0, (), km_max, False)], 0, 0
    while cola:
        _, _, g, cur, vis, rem, fin = heapq.heappop(cola)
        if fin:
            return list(vis), g, expandidos
        expandidos += 1
        prof = len(vis)
        tie += 1
        cierre = g + (m - prof)
        heapq.heappush(cola, (prio(cierre, 0.0), tie, cierre, cur, vis, rem, True))
        if prof == m:
            continue
        for i in range(1, n + 1):
            if i in vis or D[cur][i] + D[i][0] > rem:
                continue
            g2, v2 = g + costos[i - 1], vis + (i,)
            tie += 1
            heapq.heappush(cola, (prio(g2, h(v2, prof + 1)), tie, g2, i, v2, rem - D[cur][i], False))
    raise RuntimeError("cola vacia")


class AgenteVigilancia:
    """Agente basado en utilidad: percibe la semana, razona (riesgo y reglas) y actua (plan de brigadas)."""

    def __init__(self, bases, coords, th_inicio, th_cont, paradas, km_max):
        self.bases, self.xy, self.th0, self.th1, self.m, self.kmx = bases, coords, th_inicio, th_cont, paradas, km_max

    def razonar(self, f, riesgo):
        f = f.assign(riesgo=riesgo)
        return f.sort_values("riesgo", ascending=False)

    def clasificar(self, r):
        if r.en_brote == 1:
            return "Continuación de brote" if r.p >= self.th1 else "Vigilancia preventiva"
        return "Alerta de inicio" if r.p >= self.th0 else "Vigilancia preventiva"

    def planificar(self, f, dep, modo="a*"):
        base = self.xy[self.bases[dep]]
        f = f[[km(base, self.xy[u]) * 2 <= self.kmx for u in f.ubigeo]].head(N_CAND)
        if f.empty:
            return f, [], 0, 0.0
        pts = [base] + [self.xy[u] for u in f.ubigeo]
        D = [[km(a, b) for b in pts] for a in pts]
        ruta, g, exp = buscar([1 - float(x) for x in f.riesgo], D, self.m, self.kmx, modo)
        return f, ruta, exp, g


def politica(ag, te, riesgo, nombre):
    vis = np.zeros(len(te), bool)
    for (_, _, dep), f in te.assign(_r=riesgo).groupby(["ano", "semana", "departamento"], sort=False):
        o = f.sort_values("_r", ascending=False).assign(riesgo=lambda x: x._r)
        c, ruta, _, _ = ag.planificar(o, dep)
        for i in ruta:
            vis[te.index.get_loc(c.index[i - 1])] = True
    y, ini = te.y.values == 1, (te.en_brote.values == 0) & (te.y.values == 1)
    return {"politica": nombre, "visitas": int(vis.sum()), "brotes_atendidos": int((vis & y).sum()), "cobertura_brotes": float((vis & y).sum() / y.sum()),
            "inicios_atendidos": int((vis & ini).sum()), "cobertura_inicios": float((vis & ini).sum() / ini.sum()), "precision_visitas": float(y[vis].mean())}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    v2 = Path(sys.argv[1])
    m, kmx = int(sys.argv[3]) if len(sys.argv) > 3 else 3, float(sys.argv[4]) if len(sys.argv) > 4 else 400.0
    d = pd.read_pickle(v2 / "panel_v2.pkl").reset_index(drop=True)
    z = np.load(v2 / "scores_redes.npz")
    al = json.loads((v2 / "alerta_inicio.json").read_text(encoding="utf-8"))
    va, te = d[(d.ano > pl.FIN_TRAIN) & (d.ano <= pl.FIN_VALID)].copy(), d[d.ano > pl.FIN_VALID].copy()
    SCORE = pd.read_csv(v2 / "comparacion_bootstrap.csv").sort_values("valid_pr_auc", ascending=False).iloc[0].modelo.replace(" ", "_")
    print("modelo elegido en validacion:", SCORE, flush=True)
    sv, st = z[f"{SCORE}_valid"], z[f"{SCORE}_test"]
    th0 = al["modelos"][SCORE.replace("_", " ")]["umbral_inicio"]
    bv = va.en_brote.values == 1
    th1 = pl.corte_optimo(va.y.values[bv], sv[bv])

    def alertar(x, s):
        a = x.en_brote.values == 0
        return ((a & (s >= th0)) | (~a & (s >= th1))).astype(int)

    yhat = alertar(te, st)
    fin = pl.met_bin(te, yhat, st)
    ini = te.en_brote.values == 0
    fin.update({"modelo": SCORE.replace("_", " "), "umbral_inicio": th0, "umbral_continuacion": th1, "matriz_confusion": confusion_matrix(te.y, yhat).tolist(),
                "precision_inicio": float(te.y.values[ini & (yhat == 1)].mean()), "alertas_inicio": int(yhat[ini].sum()), "valid": pl.met_bin(va, alertar(va, sv), sv)})
    (v2 / "sistema_final.json").write_text(json.dumps(fin, indent=1, ensure_ascii=False), encoding="utf-8")

    te["p"], te["yhat"] = st, yhat
    alerta_reglas, te["regla"] = pl.sistema_experto(te)
    te[["ubigeo", "departamento", "provincia", "distrito", "ano", "semana", "casos", "umbral", "en_brote", "casos_sig", "umbral_sig", "y", "p", "yhat", "regla"]].to_csv(v2 / "app_panel_v2.csv.gz", index=False)

    c = pd.read_csv(sys.argv[2], dtype={"ubigeo": str}).set_index("ubigeo")
    xy = {u: (r.latitude, r.longitude) for u, r in c.iterrows()}
    bases = {dep: f"{g.ubigeo.iloc[0][:2]}0101" for dep, g in te.groupby("departamento")}
    ag = AgenteVigilancia(bases, xy, th0, th1, m, kmx)

    # plan y explicacion semana a semana
    plan, nodos = [], {"a*": 0, "ucs": 0, "voraz": 0}
    costos = {"a*": 0.0, "ucs": 0.0, "voraz": 0.0}
    no_opt, n_pr = 0, 0
    t0 = time.time()
    for (a, s, dep), f in te.groupby(["ano", "semana", "departamento"], sort=False):
        o = ag.razonar(f, f.p.values)
        cand, ruta, _, _ = ag.planificar(o, dep)
        if cand.empty:
            continue
        pts = [xy[bases[dep]]] + [xy[u] for u in cand.ubigeo]
        D = [[km(u, v) for v in pts] for u in pts]
        cs = [1 - float(x) for x in cand.riesgo]
        res = {mo: buscar(cs, D, m, kmx, mo) for mo in nodos}
        for mo, (_, g, e) in res.items():
            nodos[mo] += e; costos[mo] += g
        if abs(res["a*"][1] - res["ucs"][1]) > 1e-9:
            no_opt += 1
        n_pr += 1
        for k, i in enumerate(ruta, 1):
            r = cand.iloc[i - 1]
            plan.append({"ano": a, "semana": s, "departamento": dep, "base": bases[dep], "orden": k, "ubigeo": r.ubigeo, "distrito": r.distrito, "riesgo": float(r.p), "casos": int(r.casos), "y": int(r.y),
                         "accion": ag.clasificar(r), "regla": TEXTO_REGLA.get(r.regla, "Ninguna regla experta activa; decide el modelo."), "lat": xy[r.ubigeo][0], "lon": xy[r.ubigeo][1], "lat_base": xy[bases[dep]][0], "lon_base": xy[bases[dep]][1]})
    print("busqueda", round(time.time() - t0), "s, problemas", n_pr, "A* distinto de UCS:", no_opt, flush=True)
    pd.DataFrame(plan).to_csv(v2 / "agente_plan.csv.gz", index=False)

    rng = np.random.RandomState(pl.SEED)
    pols = [politica(ag, te, st, "Agente (modelo + A*)"), politica(ag, te, te.ratio.values / 10, "Persistencia (casos actuales)"),
            politica(ag, te, 0.9 * alerta_reglas + 0.1 * te.ratio.values / 10, "Solo reglas expertas"), politica(ag, te, te.y.values.astype(float), "Oráculo (límite teórico)")]
    al_r = [politica(ag, te, rng.rand(len(te)), "Aleatorio") for _ in range(5)]
    pols.append({k: (al_r[0][k] if k == "politica" else float(np.mean([x[k] for x in al_r]))) for k in al_r[0]})
    sens = []
    for m_ in [1, 2, 3, 5]:
        a_ = AgenteVigilancia(bases, xy, th0, th1, m_, kmx)
        sens.append({"paradas": m_, **politica(a_, te, st, "Agente"), "cobertura_persistencia": politica(a_, te, te.ratio.values / 10, "p")["cobertura_brotes"], "inicios_persistencia": politica(a_, te, te.ratio.values / 10, "p")["cobertura_inicios"]})
    res = {"escenario": {"paradas": m, "km_max": kmx, "candidatos": N_CAND, "departamentos": len(bases)}, "politicas": pols, "sensibilidad_paradas": sens,
           "busqueda": {"problemas": n_pr, "nodos_expandidos": nodos, "costo_total": costos, "a_estrella_igual_ucs": no_opt == 0, "exceso_voraz": costos["voraz"] / costos["a*"] - 1}}
    (v2 / "agente.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(pols).round(3).to_string()); print(pd.DataFrame(sens).round(3).to_string()); print(res["busqueda"])
    print(json.dumps({k: v for k, v in fin.items() if k != "valid"}, indent=0, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
