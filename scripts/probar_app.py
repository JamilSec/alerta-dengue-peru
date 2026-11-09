#!/usr/bin/env python3
"""Recorre todas las secciones de app/app.py y falla si alguna lanza excepcion."""
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest
app = Path(__file__).resolve().parent.parent / "app" / "app.py"
at = AppTest.from_file(str(app), default_timeout=120).run()
if at.exception: sys.exit(f"Inicio: {at.exception[0].message}")
opc = at.sidebar.radio[0].options
mal = 0
for o in opc:
    at.sidebar.radio[0].set_value(o).run()
    e = [x.message for x in at.exception]
    print(("OK " if not e else "ERROR ") + o, e[:1], len(at.warning))
    mal += bool(e)
sys.exit(1 if mal else 0)
