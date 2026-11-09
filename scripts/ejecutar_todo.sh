#!/bin/zsh
# Reproduce todo el proyecto. Uso: scripts/ejecutar_todo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CSV=datos/crudos/datos_abiertos_vigilancia_dengue_2000_2025_0.csv
PY=(.venv/bin/python -I)
[ -f "$CSV" ] || { [ -f "$CSV.gz" ] && gunzip -k "$CSV.gz"; }
[ -f "$CSV" ] || { echo "Falta $CSV"; exit 1; }
$PY scripts/pipeline.py "$CSV" outputs > outputs_pipeline.log 2>&1
$PY scripts/experimentos_v2.py "$CSV" datos/clima outputs/v2 > outputs/experimentos_v2.log 2>&1
$PY scripts/red_profunda.py "$CSV" datos/clima outputs/v2 25 > outputs/red_profunda.log 2>&1
$PY scripts/evaluar_v2.py outputs/v2 0.40 500 > outputs/evaluar_v2.log 2>&1
$PY scripts/agente.py outputs/v2 datos/clima/celda_por_ubigeo.csv 3 400 > outputs/agente.log 2>&1
$PY scripts/figuras_v2.py "$CSV" datos/clima outputs/v2 outputs/fig > outputs/figuras_v2.log 2>&1
echo LISTO
