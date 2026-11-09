"""Presentacion del proyecto en Streamlit. Ejecutar: streamlit run app/app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "outputs"
FIG = SAL / "fig"
V2 = SAL / "v2"

st.set_page_config(page_title="Alerta de dengue", page_icon="🦟", layout="wide")


@st.cache_data
def cargar():
    return (json.loads((SAL / "metricas.json").read_text(encoding="utf-8")),
            pd.read_csv(SAL / "comparacion_modelos.csv"),
            pd.read_csv(V2 / "app_panel_v2.csv.gz", dtype={"ubigeo": str}, keep_default_na=False, na_values=[""]),
            pd.read_csv(SAL / "clusters_distritos.csv", dtype={"ubigeo": str}),
            json.loads((V2 / "sistema_final.json").read_text(encoding="utf-8")),
            json.loads((V2 / "agente.json").read_text(encoding="utf-8")),
            pd.read_csv(V2 / "agente_plan.csv.gz", dtype={"ubigeo": str}),
            pd.read_csv(V2 / "comparacion_bootstrap.csv"),
            pd.read_csv(V2 / "variantes.csv"),
            json.loads((V2 / "analisis_gru.json").read_text(encoding="utf-8")),
            pd.read_csv(RAIZ / "datos/clima/celda_por_ubigeo.csv", dtype={"ubigeo": str}, usecols=["ubigeo", "latitude", "longitude"]))


R, COMP, PANEL, CLUST, F, AG, PLAN, BOOT, VAR, AN, COORD = cargar()
for _t, _c in [(PANEL, ["distrito", "provincia"]), (PLAN, ["distrito"])]:  # el archivo fuente perdio la Ñ
    _t[_c] = _t[_c].apply(lambda s: s.str.replace("�", "Ñ"))
PER = "-".join(map(str, R["particion"]["test"]["anos"]))
_vi = PANEL[PANEL.en_brote == 0]
AZAR_INI = float(_vi.y.mean())
pct = lambda v: f"{v:.1%}".replace(".", ",")
num = lambda v: f"{v:,.0f}".replace(",", " ")


def img(nombre, **kw):
    ruta = FIG / f"{nombre}.png"
    if ruta.exists():
        st.image(str(ruta), width="stretch", **kw)
    else:
        st.info(f"Figura pendiente de generar: {nombre}")


def regla_txt():
    return R["reglas_disparos_test"]


def portada():
    st.title("Sistema inteligente de alerta temprana de brotes de dengue por distrito en el Perú")
    st.caption("Sistemas Inteligentes y Machine Learning · Universidad Privada del Norte · Jamil Clinton Rodríguez Aguilar")
    pers = COMP[COMP.modelo.str.startswith("Regla")].iloc[0]
    st.markdown(f"### Saber que un brote sigue en curso es fácil: una regla simple logra F1 de {str(round(pers.test_f1, 2)).replace(".", ",")}. Lo difícil, y lo útil, es avisar **antes de que empiece**.")
    a, b, c, d = st.columns(4)
    a.metric("Inicios de brote anticipados", pct(F["recall_inicio_brote"]), "regla simple: 0 %", delta_color="off", delta_arrow="off")
    b.metric("Aciertos al avisar un inicio", pct(F["precision_inicio"]), f"azar: {pct(AZAR_INI)}", delta_color="off", delta_arrow="off")
    c.metric("Distritos vigilados", R["panel"]["distritos"])
    d.metric("Casos analizados", num(R["datos"]["filas_finales"]))
    st.markdown("**Pregunta:** ¿puede un sistema inteligente anticipar con una semana de antelación qué distritos superarán su umbral histórico de dengue, mejor que una regla simple?")
    img("23_ejemplo_gru")


def lista():
    st.header("La lista del lunes: lo que recibiría un equipo de vigilancia")
    st.caption("Simulación con años no vistos por el modelo: en cada semana solo se usa información disponible hasta esa semana.")
    et = PANEL.ano.astype(str) + "-S" + PANEL.semana.astype(str).str.zfill(2)
    nuevas = PANEL[(PANEL.en_brote == 0) & (PANEL.yhat == 1)].groupby(et).size()
    sem = st.select_slider("Semana", options=sorted(et.unique()), value=nuevas.idxmax())
    w = PANEL[et == sem].merge(COORD, on="ubigeo", how="left")
    ini, cont = w[(w.en_brote == 0) & (w.yhat == 1)].sort_values("p", ascending=False), w[(w.en_brote == 1) & (w.yhat == 1)]
    ver = st.toggle("Revelar qué pasó la semana siguiente", value=False)
    a, b, c = st.columns(3)
    a.metric("Alertas nuevas (inicio)", len(ini)); b.metric("Brotes que seguirían", len(cont)); c.metric("Distritos sin alerta", int((w.yhat == 0).sum()))
    if ver:
        x, y = st.columns(2)
        x.metric("De las alertas nuevas, hubo brote en", f"{int(ini.y.sum())} de {len(ini)}")
        y.metric("Inicios que se escaparon", int(((w.en_brote == 0) & (w.yhat == 0) & (w.y == 1)).sum()))
    fig = go.Figure()
    sin = w[w.yhat == 0]
    fig.add_scattermap(lon=sin.longitude, lat=sin.latitude, mode="markers", name="Sin alerta", marker=dict(size=5, color="#b8bec7"), text=sin.distrito, hovertemplate="%{text}<extra></extra>")
    fig.add_scattermap(lon=cont.longitude, lat=cont.latitude, mode="markers", name="Brote que seguiría", marker=dict(size=9, color="#3a6ea5"), text=cont.distrito + " · riesgo " + (cont.p * 100).round().astype(int).astype(str) + " %", hovertemplate="%{text}<extra></extra>")
    fig.add_scattermap(lon=ini.longitude, lat=ini.latitude, mode="markers", name="Alerta de inicio", marker=dict(size=13, color="#b5413a"), text=ini.distrito + " · riesgo " + (ini.p * 100).round().astype(int).astype(str) + " %", hovertemplate="%{text}<extra></extra>")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), map=dict(style="carto-positron", center=dict(lat=-9.3, lon=-75.5), zoom=4.6), legend=dict(orientation="h", y=1.0, x=0, bgcolor="rgba(255,255,255,.8)"))
    izq, der = st.columns([1, 1])
    with izq:
        st.plotly_chart(fig, width="stretch")
    with der:
        st.markdown("**Distritos con alerta de inicio, por riesgo**")
        motivo = {"R1_canal_superado": "Supera el canal", "R2_ascenso_sostenido": "Ascenso sostenido", "R3_foco_departamental": "Foco departamental"}
        t = pd.DataFrame({"Departamento": ini.departamento, "Distrito": ini.distrito, "Casos esta semana": ini.casos, "Riesgo": ini.p.map(lambda v: f"{v:.0%}"), "Motivo": ini.regla.map(motivo).fillna("Patrón aprendido por el modelo")})
        if ver:
            t["Semana siguiente"] = ini.y.map({1: "Brote", 0: "Sin brote"})
        st.dataframe(t, hide_index=True, width="stretch", height=470)
    c_ = AN["carga"]
    st.caption(f"Carga típica en 2023-2025: {c_['alertas_semana']['mediana']:.0f} alertas por semana en todo el país, {c_['nuevas_semana']['mediana']:.0f} de ellas nuevas; en la semana más intensa, {c_['alertas_semana']['maximo']}. Mapa: © OpenStreetMap, © CARTO.")


def problema():
    st.header("El problema")
    izq, der = st.columns([1, 1])
    izq.markdown("""
- El dengue llega en epidemias recurrentes y las respuestas locales son limitadas: fumigación, personal y camas.
- Si la alerta surge cuando el brote ya es visible, se pierde tiempo.
- **Usuario:** equipos de vigilancia epidemiológica (MINSA, CDC Perú, DIRESA).
- **Salida del sistema:** lista semanal de distritos priorizados y motivo de cada alerta.
""")
    with der:
        img("03_estacionalidad")


def datos():
    st.header("Datos y preparación")
    d = R["datos"]
    embudo = pd.DataFrame({"Etapa": ["Filas crudas", "Con claves válidas", "Categorías válidas", "Final"],
                           "Filas": [d["filas_crudas"], d["tras_nulos_clave"], d["tras_validar_categorias"], d["filas_finales"]]})
    a, b = st.columns(2)
    with a:
        st.markdown(f"Dataset abierto de vigilancia del MINSA/CDC Perú, años {d['anos'][0]}-{d['anos'][1]}, {d['departamentos']} departamentos.")
        st.dataframe(embudo, hide_index=True, width="stretch")
        st.markdown(f"**Panel distrito-semana:** {num(R['panel']['filas'])} filas, {R['panel']['n_features']} variables, prevalencia de brote {pct(R['panel']['prevalencia'])}.")
    with b:
        img("01_casos_por_ano")
    st.subheader("Definición de brote")
    st.latex(r"\text{brote}_{d,t} = 1 \iff \text{casos}_{d,t} > \max\left(\mu_{d,s} + 2\sigma_{d,s},\ 5\right)")
    st.caption("μ y σ: media y desviación de la misma semana en los 5 años previos del distrito. Solo usa información pasada.")


def senal():
    st.header("La serie semanal como señal")
    s = R["senales"]
    st.markdown(f"Se detectaron **{s['n_picos']} picos** en la serie nacional, suavizada con Savitzky-Golay. Años con picos: {', '.join(map(str, s['anos_picos']))}.")
    a, b = st.columns(2)
    with a:
        img("06_senal")
    with b:
        img("07_periodograma")


def diseno():
    st.header("Diseño experimental")
    p = R["particion"]
    st.dataframe(pd.DataFrame([{"Conjunto": k, "Años": f"{v['anos'][0]}-{v['anos'][1]}", "Filas": v["n"], "Prevalencia": pct(v["prevalencia"])} for k, v in p.items()]), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.markdown("""
**Sin fuga de datos**
- Entrenar con el pasado y probar con años futuros.
- El umbral se calcula con años previos.
- Las variables solo usan la semana actual o anteriores.
""")
    m = R["mejoras"]
    b.markdown(f"""
**¿Por qué no una partición aleatoria?**
- Aleatoria: PR-AUC **{m['particion_aleatoria_pr_auc']:.3f}**
- Temporal: PR-AUC **{m['particion_temporal_pr_auc']:.3f}**

La partición aleatoria mezcla semanas futuras en el entrenamiento y mide algo distinto al uso real.
""")


def experto():
    st.header("Sistema experto: la línea base simbólica")
    st.markdown("""
| Regla | Condición | Explicación entregada |
|---|---|---|
| R1 | Casos actuales > canal endémico | El brote ya está en curso |
| R2 | Dos semanas de ascenso y casos sobre la media de la próxima semana | Ascenso sostenido |
| R3 | ≥2 distritos del departamento en brote y distrito cerca de su umbral | Foco departamental |
""")
    t = COMP[COMP.modelo.str.startswith(("Regla", "Sistema"))][["modelo", "test_precision", "test_recall", "test_f1", "test_recall_inicio"]]
    st.dataframe(t.round(3), hide_index=True, width="stretch")
    st.caption("Recall de inicio: brotes nuevos detectados, es decir, cuando el distrito aún no estaba en brote. Es la parte difícil del problema.")


def modelos():
    st.header(f"Modelos clásicos (prueba {PER})")
    cols = ["modelo", "valid_pr_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1", "test_recall_inicio"]
    st.dataframe(COMP[cols].round(3).sort_values("test_f1", ascending=False), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        img("09_comparacion")
    with b:
        img("08_curvas_pr")
    st.success(f"Mejor modelo clásico por PR-AUC en validación, no en prueba: **{R['seleccion']['modelo']}**")


def final():
    st.header(f"Sistema final: {F['modelo']}")
    a, b, c, d = st.columns(4)
    a.metric("Precisión", f"{F['precision']:.3f}"); b.metric("Recall", f"{F['recall']:.3f}"); c.metric("F1", f"{F['f1']:.3f}"); d.metric("Recall de inicio", f"{F['recall_inicio_brote']:.3f}")
    i, j = st.columns(2)
    e = AN["errores"]; tr = AN["entrenamiento"]
    with i:
        cm = F["matriz_confusion"]
        st.dataframe(pd.DataFrame({"": ["Real: sin brote", "Real: brote"], "Predicho: sin brote": [cm[0][0], cm[1][0]], "Predicho: brote": [cm[0][1], cm[1][1]]}), hide_index=True, width="stretch")
        st.markdown(f"**Alerta de inicio:** detecta **{pct(F['recall_inicio_brote'])}** de los {F['n_inicios']} inicios con precisión **{pct(F['precision_inicio'])}**; el azar sería {pct(AZAR_INI)}. Umbrales fijados en validación.")
        st.markdown(f"**Sin sobreajuste:** F1 {tr['f1']:.3f} en entrenamiento, {F['valid']['f1']:.3f} en validación y {F['f1']:.3f} en prueba.")
        st.markdown(f"**Dónde falla:** {pct(e['FN_que_son_inicio'])} de los brotes omitidos son inicios. El distrito tenía {e['FN_media_casos_semana_actual']:.1f} casos de media la semana previa, frente a {e['VP_media_casos_semana_actual']:.1f} en los detectados.")
    with j:
        img("22_importancia_gru")
        st.caption("Permutación de grupos de entradas sobre la propia GRU, en prueba.")
    fin_v = R["particion"]["valid"]["anos"][1]
    v = R["validacion_temporal"]
    with st.expander(f"Validación temporal expansiva {fin_v - 6}-{fin_v} de los modelos clásicos"):
        st.markdown(" · ".join(f"{k}: PR-AUC {x['media']:.3f} ± {x['de']:.3f}" for k, x in v.items()))


def robustez():
    st.header("¿Se sostiene?")
    st.subheader("Año por año")
    a = pd.DataFrame(AN["por_ano"])
    img("24_por_ano")
    st.markdown(f"El PR-AUC global depende de cuántos brotes hubo: {a.pr_auc.iloc[0]:.2f} en la epidemia de {a.ano.iloc[0]} y {a.pr_auc.iloc[-1]:.2f} en {a.ano.iloc[-1]}, con solo {pct(a.prevalencia.iloc[-1])} de semanas en brote. **La detección de inicios se mantiene entre {pct(a.recall_inicio.min())} y {pct(a.recall_inicio.max())}**, con precisión de {pct(a.precision_inicio.min())} a {pct(a.precision_inicio.max())}.")
    st.subheader("¿Una semana alcanza?")
    img("26_horizontes")
    h = pd.DataFrame(AN["horizontes"])
    h = h[h.modelo != "Persistencia"][["horizonte", "modelo", "f1", "pr_auc_inicio", "recall_inicio", "precision_inicio", "azar_inicio"]].round(3).rename(columns={"horizonte": "Semanas"})
    st.dataframe(h, hide_index=True, width="stretch")
    H = pd.DataFrame(AN["horizontes"]); g1, g4 = [H[(H.modelo == "GRU") & (H.horizonte == k)].iloc[0] for k in (1, 4)]; b4 = H[(H.modelo == "Gradient Boosting") & (H.horizonte == 4)].iloc[0]
    st.markdown(f"Lo global cae (PR-AUC de {g1.pr_auc:.3f} a {g4.pr_auc:.3f}), pero **a 4 semanas la GRU aún detecta {pct(g4.recall_inicio)} de los inicios con precisión {pct(g4.precision_inicio)}**. Gradient Boosting se queda en {pct(b4.recall_inicio)}: la historia de 52 semanas pesa más cuanto más lejos se predice.")
    st.subheader("¿Cuánto trabajo genera?")
    img("25_carga_alertas")
    c_, ti = AN["carga"], AN["tamano_inicios"]
    st.markdown(f"Mediana de **{c_['nuevas_semana']['mediana']:.0f} alertas nuevas por semana** en todo el país. De los 10 distritos de mayor riesgo que aún no están en brote, {pct(c_['top10_precision'])} entra en brote la semana siguiente (azar {pct(c_['azar_inicio'])}).")
    st.markdown(f"Los inicios detectados concentran **{pct(ti['proporcion_casos_en_detectados'])}** de los casos de las {ti['ventana_semanas']} semanas siguientes a un inicio (mediana {ti['mediana_detectados']:.0f} casos frente a {ti['mediana_omitidos']:.0f} en los omitidos).")


def no_funciono():
    st.header("Lo que no funcionó, y por qué importa")
    st.subheader("Agregar clima")
    st.markdown("Se añadieron lluvia, temperatura y humedad de NASA POWER, solo con información hasta la semana actual, y un modelo de dos etapas (distrito sin brote o en brote).")
    t = VAR[["variante", "valid_pr_auc", "test_pr_auc", "test_f1", "test_pr_auc_inicio", "test_recall_inicio", "test_precision_inicio"]].round(3)
    st.dataframe(t, hide_index=True, width="stretch")
    st.caption("Las variantes se comparan en validación. La prueba se reporta para contrastar y no se usó para elegir. La importancia en la GRU confirma el resultado: permutar el clima casi no cambia sus predicciones.")
    st.subheader("Predecir severidad clínica")
    s, t = R["severidad_binaria"], R["severidad_3clases"]
    a, b, c = st.columns(3)
    a.metric("ROC-AUC (con señales/grave)", f"{s['roc_auc']:.3f}"); b.metric("PR-AUC", f"{s['pr_auc']:.3f}", f"azar {s['prevalencia']:.3f}", delta_color="off"); c.metric("F1 macro, 3 clases", f"{t['f1_macro']:.3f}")
    st.markdown("El dataset no incluye signos clínicos, y las categorías actuales casi no existen antes de 2010. Por eso el proyecto se centró en brotes.")
    img("04_severidad_ano")


def redes():
    st.header("Redes neuronales e imagen histórica")
    a, b = st.columns(2)
    with a:
        img("16_imagen_cnn")
        st.markdown("Cada caso se convierte en una **imagen** de 6 años por 32 semanas, con casos y lluvia como canales. La **CNN** la clasifica; la **GRU** lee las últimas 52 semanas.")
    with b:
        img("17_curvas_redes")
    i, j = st.columns(2)
    with i:
        img("18_bootstrap")
    with j:
        st.dataframe(BOOT[["modelo", "valid_pr_auc", "test_pr_auc", "ic95_inf", "ic95_sup", "test_pr_auc_inicio"]].round(3), hide_index=True, width="stretch")


def agente():
    st.header("Agente de brigadas: de la alerta a la decisión")
    e = AG["escenario"]; b = AG["busqueda"]
    st.markdown(f"Cada semana y por departamento, **una brigada** sale de la capital y visita hasta **{e['paradas']} distritos** con un recorrido máximo de **{e['km_max']:.0f} km** en línea recta. El agente **percibe** los datos, **razona** con el modelo y las reglas, y **actúa** eligiendo la ruta con A*.")
    a, c = st.columns([3, 2])
    with a:
        img("19_agente_cobertura")
    with c:
        n = b["nodos_expandidos"]
        st.metric("Nodos que expande A*", num(n["a*"]), f"sin heurística: {num(n['ucs'])}", delta_color="off")
        st.metric("Solución óptima", "Sí" if b["a_estrella_igual_ucs"] else "No")
        st.metric("Costo extra de la búsqueda voraz", f"+{b['exceso_voraz']:.1%}")
    st.subheader("Plan semanal")
    et = (PLAN.ano.astype(str) + "-S" + PLAN.semana.astype(str).str.zfill(2))
    op = sorted(et.unique())
    pico = PLAN.groupby(et).y.sum().idxmax()
    sem = st.select_slider("Semana", options=op, value=pico)
    w = PLAN[et == sem].sort_values(["departamento", "orden"])
    fig = go.Figure()
    lon_r, lat_r = [], []
    for _, g in w.groupby("departamento"):
        lon_r += [g.lon_base.iloc[0], *g.lon, g.lon_base.iloc[0], None]
        lat_r += [g.lat_base.iloc[0], *g.lat, g.lat_base.iloc[0], None]
    fig.add_scattermap(lon=lon_r, lat=lat_r, mode="lines", name="Ruta de brigada", line=dict(color="#5f6b7a", width=2), hoverinfo="skip")
    bases = w[["lon_base", "lat_base", "departamento"]].drop_duplicates()
    fig.add_scattermap(lon=bases.lon_base, lat=bases.lat_base, mode="markers", name="Base de brigada", marker=dict(size=9, color="#222222"), text="Base: " + bases.departamento, hovertemplate="%{text}<extra></extra>")
    for v, nom, col in [(1, "Visita: hubo brote", "#b5413a"), (0, "Visita: sin brote", "#3a6ea5")]:
        g = w[w.y == v]
        fig.add_scattermap(lon=g.lon, lat=g.lat, mode="markers", name=nom, marker=dict(size=12, color=col), text=g.distrito + " (" + g.departamento + ") · riesgo " + (g.riesgo * 100).round().astype(int).astype(str) + " %", hovertemplate="%{text}<extra></extra>")
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), map=dict(style="carto-positron", center=dict(lat=-9.3, lon=-75.5), zoom=4.8), legend=dict(orientation="h", y=1.0, x=0, bgcolor="rgba(255,255,255,.8)"))
    x, y = st.columns([1, 1])
    with x:
        st.plotly_chart(fig, width="stretch")
    with y:
        t = w[["departamento", "orden", "distrito", "riesgo", "accion", "y"]].rename(columns={"y": "brote real"}).copy()
        t["riesgo"] = t.riesgo.round(2)
        st.dataframe(t, hide_index=True, width="stretch", height=430)
        d = st.selectbox("Explicación de la visita", w.distrito.tolist())
        f = w[w.distrito == d].iloc[0]
        st.caption(f"{f.distrito}: {f.accion}. Riesgo {f.riesgo:.0%}. Regla: {f.regla}")
    st.caption("Mapa: © OpenStreetMap, © CARTO. Requiere internet para el fondo. Los km son línea recta entre centros de distrito: no incluyen carreteras reales.")


def demo():
    st.header(f"Demo en vivo: alerta por distrito ({PER})")
    a, b, c = st.columns(3)
    dep = a.selectbox("Departamento", sorted(PANEL.departamento.unique()), index=sorted(PANEL.departamento.unique()).index("PIURA") if "PIURA" in set(PANEL.departamento) else 0)
    sub = PANEL[PANEL.departamento == dep]
    orden = sub.groupby("distrito").y.sum().sort_values(ascending=False).index.tolist()
    dis = b.selectbox("Distrito", orden)
    g = sub[sub.distrito == dis].sort_values(["ano", "semana"]).reset_index(drop=True)
    et = g.ano.astype(str) + "-S" + g.semana.astype(str).str.zfill(2)
    sem = c.select_slider("Semana de análisis", options=et.tolist(), value=et.iloc[int(g.p.idxmax())])
    i = et.tolist().index(sem)
    fig = go.Figure()
    fig.add_scatter(x=et, y=g.casos_sig, name="Casos (semana siguiente)", line=dict(color="#333"))
    fig.add_scatter(x=et, y=g.umbral_sig, name="Umbral", line=dict(color="#e0a030", dash="dash"))
    al = g[g.yhat == 1]
    fig.add_scatter(x=et[al.index], y=al.casos_sig, mode="markers", name="Alerta", marker=dict(color="#b5413a", size=8))
    fig.add_vline(x=i, line_color="#3a6ea5")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12), xaxis=dict(nticks=16, tickangle=-45))
    st.plotly_chart(fig, width="stretch")
    f = g.iloc[i]
    x, y, z = st.columns(3)
    x.metric("Probabilidad de brote la próxima semana", pct(f.p))
    y.metric("Decisión", "ALERTA" if f.yhat else "Sin alerta")
    z.metric("Lo que pasó", "Brote" if f.y else "No brote")
    reglas = {"R1_canal_superado": "R1: ya supera el canal endémico.", "R2_ascenso_sostenido": "R2: ascenso sostenido.", "R3_foco_departamental": "R3: foco departamental."}
    st.caption("Sistema experto: " + reglas.get(f.regla, "ninguna regla activa.") if isinstance(f.regla, str) else "Sistema experto: ninguna regla activa.")


def perfiles():
    st.header("Perfiles de distritos (no supervisado)")
    u = R["no_supervisado"]
    a, b = st.columns(2)
    with a:
        img("14_clusters")
    with b:
        img("15_silhouette")
    st.markdown(f"K-Means con **k={u['k_elegido']}** por silhouette; PCA explica {pct(sum(u['pca_varianza']))} de la varianza en 2 componentes." + (f" DBSCAN: {u['dbscan']['clusters']} grupos, {pct(u['dbscan']['ruido'])} de ruido." if u["dbscan"] else ""))
    st.dataframe(pd.DataFrame(u["perfil_clusters"]).round(2), hide_index=True, width="stretch")


def cierre():
    st.header("Conclusiones")
    pers = COMP[COMP.modelo.str.startswith("Regla")].iloc[0]
    pol = {p["politica"]: p for p in AG["politicas"]}
    ag_, pe = pol["Agente (modelo + A*)"], pol["Persistencia (casos actuales)"]
    A_ = pd.DataFrame(AN["por_ano"]); H = pd.DataFrame(AN["horizontes"]); g4 = H[(H.modelo == "GRU") & (H.horizonte == 4)].iloc[0]
    a, b = st.columns(2)
    a.markdown(f"""
**Hallazgos**
- Anticipa **{pct(F['recall_inicio_brote'])}** de los inicios de brote con precisión **{pct(F['precision_inicio'])}**; la regla simple no anticipa ninguno.
- Esa detección es estable: de {pct(A_.recall_inicio.min())} a {pct(A_.recall_inicio.max())} en cada año, con muchos o pocos brotes.
- A 4 semanas aún detecta **{pct(g4.recall_inicio)}** con precisión **{pct(g4.precision_inicio)}**.
- El F1 global apenas supera a la persistencia ({F['f1']:.3f} frente a {pers.test_f1:.3f}), y el PR-AUC baja a {A_.pr_auc.iloc[-1]:.2f} en el año con menos brotes.
- El agente atiende **{pct(ag_['cobertura_inicios'])}** de los inicios con una brigada por departamento, frente a **{pct(pe['cobertura_inicios'])}** de la persistencia.
- El clima no mejoró la predicción.
""")
    b.markdown("""
**Limitaciones y ética**
- Clima en malla de unos 50 km y sin datos entomológicos.
- No modela retrasos de notificación.
- Aún no detecta cerca de la mitad de los inicios, sobre todo los que arrancan con pocos casos.
- Las redes se entrenaron con una sola semilla.
- Las rutas usan distancias en línea recta y capacidad supuesta.
- No sustituye el juicio epidemiológico: siempre con supervisión humana.
- Cuasi-identificadores: solo se muestran agregados por distrito.
""")


SEC = {"Portada": portada, "La lista del lunes": lista, "Problema": problema, "Datos": datos, "Señal": senal, "Perfiles de distritos": perfiles, "Diseño experimental": diseno, "Sistema experto": experto, "Modelos clásicos": modelos, "Redes e imagen": redes, "Sistema final": final, "¿Se sostiene?": robustez, "Lo que no funcionó": no_funciono, "Agente de brigadas": agente, "Conclusiones": cierre, "Demo en vivo": demo}
nombres = list(SEC)
st.session_state.setdefault("sec", nombres[0])


def mover_seccion(paso):
    st.session_state.sec = nombres[nombres.index(st.session_state.sec) + paso]


with st.sidebar:
    st.markdown("### Presentación")
    st.radio("Sección", nombres, key="sec", label_visibility="collapsed")
i = nombres.index(st.session_state.sec)
SEC[nombres[i]]()
st.divider()
a, _, c = st.columns([1, 6, 1])
a.button("← Anterior", disabled=i == 0, on_click=mover_seccion, args=(-1,))
c.button("Siguiente →", disabled=i == len(nombres) - 1, on_click=mover_seccion, args=(1,))
