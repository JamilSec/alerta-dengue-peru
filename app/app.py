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
            pd.read_csv(V2 / "variantes.csv"))


R, COMP, PANEL, CLUST, F, AG, PLAN, BOOT, VAR = cargar()
PER = "-".join(map(str, R["particion"]["test"]["anos"]))
_vi = PANEL[PANEL.en_brote == 0]
AZAR_INI = float(_vi.y.mean())
pct = lambda v: f"{v:.1%}"
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
    a, b, c, d = st.columns(4)
    a.metric("Casos analizados", num(R["datos"]["filas_finales"]))
    b.metric("Distritos en el panel", R["panel"]["distritos"])
    c.metric(f"PR-AUC en {PER}", f"{F['pr_auc']:.3f}", f"azar {F['prevalencia']:.3f}", delta_color="off")
    d.metric(f"F1 en {PER}", f"{F['f1']:.3f}")
    st.markdown("**Pregunta:** ¿puede un sistema inteligente anticipar con una semana de antelación qué distritos superarán su umbral histórico de dengue, mejor que una regla simple?")
    img("12_ejemplo_distrito")


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
    with i:
        cm = F["matriz_confusion"]
        st.dataframe(pd.DataFrame({"": ["Real: sin brote", "Real: brote"], "Predicho: sin brote": [cm[0][0], cm[1][0]], "Predicho: brote": [cm[0][1], cm[1][1]]}), hide_index=True, width="stretch")
        st.markdown(f"Alerta de inicio: umbral fijado en validación. Detecta **{pct(F['recall_inicio_brote'])}** de los {F['n_inicios']} inicios con precisión **{pct(F['precision_inicio'])}**; el azar sería {pct(AZAR_INI)}.")
    with j:
        img("11_importancia")
    v = R["validacion_temporal"]
    st.markdown("**Validación temporal expansiva (2014-2020):** " + " · ".join(f"{k}: PR-AUC {x['media']:.3f} ± {x['de']:.3f}" for k, x in v.items()))


def clima():
    st.header("¿Ayudan el clima y un modelo para inicios?")
    st.markdown("Se añadieron lluvia, temperatura y humedad de NASA POWER, solo con información hasta la semana actual, y un modelo de dos etapas (distrito sin brote o en brote).")
    t = VAR[["variante", "valid_pr_auc", "test_pr_auc", "test_f1", "test_pr_auc_inicio", "test_recall_inicio", "test_precision_inicio"]].round(3)
    st.dataframe(t, hide_index=True, width="stretch")
    st.caption("Las variantes se comparan en validación. La prueba se reporta para contrastar y no se usó para elegir.")


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
    st.header(f"Demo: alerta por distrito ({PER})")
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


def severidad():
    st.header("Lo que no funcionó: predecir severidad")
    s, t = R["severidad_binaria"], R["severidad_3clases"]
    a, b, c = st.columns(3)
    a.metric("ROC-AUC (con señales/grave)", f"{s['roc_auc']:.3f}"); b.metric("PR-AUC", f"{s['pr_auc']:.3f}", f"azar {s['prevalencia']:.3f}", delta_color="off"); c.metric("F1 macro, 3 clases", f"{t['f1_macro']:.3f}")
    st.markdown("El dataset no incluye signos clínicos, y las categorías actuales casi no existen antes de 2010. Por eso el proyecto se centró en brotes.")
    img("04_severidad_ano")


def cierre():
    st.header("Conclusiones")
    pers = COMP[COMP.modelo.str.startswith("Regla")].iloc[0]
    pol = {p["politica"]: p for p in AG["politicas"]}
    ag_, pe = pol["Agente (modelo + A*)"], pol["Persistencia (casos actuales)"]
    a, b = st.columns(2)
    a.markdown(f"""
**Hallazgos**
- F1 en prueba: **{F['f1']:.3f}**, frente a **{pers.test_f1:.3f}** de la regla de persistencia.
- Inicios de brote detectados: **{pct(F['recall_inicio_brote'])}**, con precisión **{pct(F['precision_inicio'])}**; la regla no detecta ninguno.
- El agente atiende **{pct(ag_['cobertura_inicios'])}** de los inicios con una brigada por departamento, frente a **{pct(pe['cobertura_inicios'])}** de la persistencia.
- El clima no mejoró la predicción, y las redes solo mejoran un poco al mejor modelo tabular.
""")
    b.markdown("""
**Limitaciones y ética**
- Clima en malla de unos 50 km y sin datos entomológicos.
- No modela retrasos de notificación.
- Las rutas usan distancias en línea recta y capacidad supuesta.
- Las epidemias de 2023 y 2024 elevan los umbrales posteriores.
- No sustituye el juicio epidemiológico: siempre con supervisión humana.
- Cuasi-identificadores: solo se muestran agregados por distrito.
""")


SEC = {"Portada": portada, "Problema": problema, "Datos": datos, "Señal": senal, "Diseño experimental": diseno, "Sistema experto": experto, "Modelos clásicos": modelos, "Clima e inicios": clima, "Redes e imagen": redes, "Sistema final": final, "Agente de brigadas": agente, "Demo por distrito": demo, "Perfiles de distritos": perfiles, "Severidad": severidad, "Conclusiones": cierre}
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
