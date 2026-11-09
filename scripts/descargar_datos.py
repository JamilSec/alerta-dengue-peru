#!/usr/bin/env python3
"""
Descarga y verifica los datasets del proyecto (dataset principal: vigilancia de
dengue MINSA/CDC Peru; secundario: fallecidos COVID-19 MINSA).

Uso:
    python3 descargar_datos.py [--dest RUTA] [--solo-dengue]

No requiere librerias externas (solo stdlib). Verificado 2026-11-09.
Guarda los CSV crudos en RUTA/ (por defecto ./datos/crudos) y imprime un
resumen de verificacion (tamano, filas, columnas, variable objetivo).
"""
import argparse
import csv
import hashlib
import os
import sys
import urllib.request
from collections import Counter

FUENTES = {
    # id: (url, nombre_archivo, descripcion)
    "dengue": (
        "https://www.datosabiertos.gob.pe/sites/default/files/datos_abiertos_vigilancia_dengue.csv",
        "dengue_minsa.csv",
        "Vigilancia epidemiologica de dengue (MINSA/CDC Peru). Objetivo: columna 'enfermedad' (severidad).",
    ),
    "fallecidos": (
        "https://cloud.minsa.gob.pe/s/xJ2LQ3QyRW38Pe5/download",
        "fallecidos_covid_minsa.csv",
        "Fallecidos por COVID-19 (MINSA). Solo defunciones; sin objetivo binario nativo.",
    ),
}

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")


def descargar(url: str, destino: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as r, open(destino, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    if total == 0:
        raise RuntimeError(f"Descarga vacia: {url}")


def verificar(ruta: str) -> None:
    size = os.path.getsize(ruta)
    with open(ruta, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    with open(ruta, "r", encoding="utf-8", errors="replace", newline="") as f:
        muestra = f.readline()
        delim = ";" if muestra.count(";") > muestra.count(",") else ","
        f.seek(0)
        rd = csv.reader(f, delimiter=delim)
        cols = next(rd)
        n = 0
        for _ in rd:
            n += 1
    print(f"  archivo        : {ruta}")
    print(f"  tamano (bytes) : {size:,}")
    print(f"  md5            : {md5}")
    print(f"  filas de datos : {n:,}")
    print(f"  columnas ({len(cols)}): {cols}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join("datos", "crudos"))
    ap.add_argument("--solo-dengue", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    ids = ["dengue"] if args.solo_dengue else list(FUENTES)
    rc = 0
    for i in ids:
        url, nombre, desc = FUENTES[i]
        destino = os.path.join(args.dest, nombre)
        print(f"\n[{i}] {desc}\n  url: {url}")
        try:
            descargar(url, destino)
            verificar(destino)
        except Exception as e:  # pragma: no cover
            print(f"  ERROR: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
