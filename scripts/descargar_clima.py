#!/usr/bin/env python3
"""Descarga clima diario de NASA POWER por celda de malla para los distritos del Peru.

Uso: descargar_clima.py <ubigeo_distrito.csv> <dir_salida> [hilos]
Es reanudable: las celdas ya guardadas no se vuelven a pedir.
Salida: <dir_salida>/celdas/<lat>_<lon>.csv y <dir_salida>/celda_por_ubigeo.csv
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX", "RH2M"]
INICIO, FIN = "20000101", "20251231"
LAT_PASO, LON_PASO = 0.5, 0.625  # malla MERRA-2


def celda(lat, lon):
    return round(round(lat / LAT_PASO) * LAT_PASO, 3), round(round(lon / LON_PASO) * LON_PASO, 3)


def pedir(lat, lon, destino, intentos=5):
    q = urllib.parse.urlencode({"parameters": ",".join(PARAMS), "community": "AG", "latitude": lat, "longitude": lon,
                                "start": INICIO, "end": FIN, "format": "JSON"})
    for i in range(intentos):
        try:
            with urllib.request.urlopen(f"{URL}?{q}", timeout=120) as r:
                datos = json.load(r)["properties"]["parameter"]
            d = pd.DataFrame(datos).rename_axis("fecha").reset_index().replace(-999.0, pd.NA)
            d.to_csv(destino, index=False)
            return len(d)
        except Exception as e:
            ultimo = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"celda {lat},{lon}: {ultimo}")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    u = pd.read_csv(sys.argv[1], dtype={"inei": str})
    prov = u.inei.str[:4]
    sin_coord = u.latitude.isna() | u.longitude.isna()
    for col in ["latitude", "longitude", "altitude"]:  # distritos nuevos sin coordenadas: promedio de su provincia
        u[col] = u[col].fillna(u.groupby(prov)[col].transform("mean"))
    print("sin coordenadas, imputados por provincia:", u.inei[sin_coord].tolist(), flush=True)
    u = u.dropna(subset=["latitude", "longitude"])
    sal = Path(sys.argv[2])
    (sal / "celdas").mkdir(parents=True, exist_ok=True)
    hilos = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    c = [celda(a, o) for a, o in zip(u.latitude, u.longitude)]
    u["lat_c"], u["lon_c"] = [x[0] for x in c], [x[1] for x in c]
    u.rename(columns={"inei": "ubigeo"})[["ubigeo", "lat_c", "lon_c", "latitude", "longitude", "altitude"]].to_csv(sal / "celda_por_ubigeo.csv", index=False)
    celdas = sorted(set(c))
    faltan = [(a, o) for a, o in celdas if not (sal / "celdas" / f"{a}_{o}.csv").exists()]
    print(f"celdas={len(celdas)} faltan={len(faltan)}", flush=True)
    errores = 0
    with ThreadPoolExecutor(hilos) as ex:
        fut = {ex.submit(pedir, a, o, sal / "celdas" / f"{a}_{o}.csv"): (a, o) for a, o in faltan}
        for n, f in enumerate(as_completed(fut), 1):
            try:
                f.result()
            except Exception as e:
                errores += 1
                print("ERROR", e, flush=True)
            if n % 25 == 0:
                print(f"{n}/{len(faltan)}", flush=True)
    print(f"FIN errores={errores} guardadas={len(list((sal / 'celdas').glob('*.csv')))}", flush=True)
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
