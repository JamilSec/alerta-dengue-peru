# Alerta temprana de brotes de dengue por distrito en el Perú

Proyecto final del curso Sistemas Inteligentes y Machine Learning (UPN). Predice, con una semana de anticipación, si un distrito entrará o seguirá en brote, y usa esa alerta para planificar visitas de brigadas.

## Resultados en prueba (2023-2025)

| Sistema | PR-AUC | F1 | Recall de inicios |
|---|---|---|---|
| GRU (elegido en validación) | 0,914 | 0,806 | 49,0 % |
| Gradient Boosting ajustado | 0,906 | 0,816 | 24,6 % |
| Sistema experto (reglas) | — | 0,722 | 57,3 % |
| Persistencia | — | 0,794 | 0 % |

Partición temporal: entrenamiento hasta 2020, validación 2021-2022, prueba 2023-2025. El detalle está en `entrega/Informe_Final_SISIN.docx`.

## Ejecutar la presentación

Los resultados ya están en `outputs/`, así que la app no necesita reentrenar.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/app.py
```

## Reproducir todo

```bash
scripts/ejecutar_todo.sh
```

Descomprime el CSV si hace falta y corre, en orden, el pipeline, los experimentos con clima, las redes (CNN y GRU, usa GPU Apple MPS si existe), la evaluación, el agente A* y las figuras.

## Estructura

- `app/`: presentación en Streamlit.
- `scripts/`: pipeline, modelos, agente y figuras.
- `outputs/`: métricas, modelos entrenados y figuras.
- `datos/`: dengue (comprimido), clima y ubigeos.
- `entrega/`: informe final.
- `notas/referencias.md`: fuentes consultadas.

## Datos

- Vigilancia epidemiológica de dengue, MINSA, Plataforma Nacional de Datos Abiertos (ODC-BY).
- Clima: NASA POWER (MERRA-2).
- Ubigeos: `jmcastagnetto/ubigeo-peru-aumentado`.
