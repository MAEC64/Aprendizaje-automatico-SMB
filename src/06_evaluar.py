"""Paso 6: compara los agentes A y B entre sí y contra la curva humana.

Uso (después de entrenar al menos una semilla de cada agente):
    python src/06_evaluar.py
Resultados (en resultados/):
    curvas_aprendizaje.png, agentes_vs_humano.png, comparacion_agentes.csv
"""
import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import comun

R = comun.RESULTADOS
evals = []
for ruta in sorted(R.glob("agente_*_s*_evaluacion.csv")):
    agente, semilla = re.match(r"agente_(\w)_s(\d+)_", ruta.name).groups()
    evals.append(pd.read_csv(ruta).assign(agente=agente, semilla=int(semilla)))
if not evals:
    raise SystemExit("No hay resultados de entrenamiento. Corre primero 05_entrenar_agente.py")
evals = pd.concat(evals)

# 1) Curvas de aprendizaje: promedio de semillas, con banda min-max
fig, ax = plt.subplots(figsize=(9, 5))
for agente, g in evals.groupby("agente"):
    curva = g.groupby("episodio").x_max_media.agg(["mean", "min", "max"])
    etiqueta = "A (desde cero)" if agente == "A" else "B (tabla Q humana)"
    ax.plot(curva.index, curva["mean"], lw=2, label=etiqueta)
    ax.fill_between(curva.index, curva["min"], curva["max"], alpha=0.2)
ax.axhline(comun.MITAD_NIVEL, ls="--", color="gray", lw=1, label="mitad del nivel")
ax.set(xlabel="Episodios de entrenamiento", ylabel="Distancia máxima media en evaluación (px)",
       title="Curvas de aprendizaje (evaluación sin exploración)")
ax.legend()
fig.savefig(R / "curvas_aprendizaje.png", dpi=120, bbox_inches="tight")

# 2) Tabla comparativa al final del entrenamiento
filas = []
for (agente, semilla), g in evals.groupby(["agente", "semilla"]):
    g = g.sort_values("episodio")
    llega = g[g.x_max_media >= comun.MITAD_NIVEL]
    filas.append({"agente": agente, "semilla": semilla,
                  "x_max_media_final": g.x_max_media.iloc[-1],
                  "completados_final": g.completados.iloc[-1],
                  "episodios_hasta_mitad": llega.episodio.iloc[0] if len(llega) else np.nan})
tabla = pd.DataFrame(filas)

# 3) Comparación con la curva humana (regresión del paso 3)
ruta_curva = R / "curva_humana.joblib"
if ruta_curva.exists():
    curva = joblib.load(ruta_curva)
    fig, ax = plt.subplots(figsize=(9, 5))
    t_max = 0.0
    diferencias = []
    for ruta in sorted(R.glob("agente_*_s*_trayectoria.csv")):
        tray = pd.read_csv(ruta)
        if tray.empty:
            continue
        agente, semilla = re.match(r"agente_(\w)_s(\d+)_", ruta.name).groups()
        humano = curva["modelo"].predict(tray[["t"]].to_numpy())
        diferencias.append({"agente": agente, "semilla": int(semilla),
                            "dif_media_vs_humano_px": float(np.mean(tray.x - humano))})
        ax.plot(tray.t, tray.x, lw=1.5, alpha=0.8, label=f"agente {agente}, semilla {semilla}")
        t_max = max(t_max, tray.t.max())
    t = np.linspace(0, max(t_max, 1), 300).reshape(-1, 1)
    ax.plot(t, curva["modelo"].predict(t), "k--", lw=2.5, label=f"humano ({curva['nombre']})")
    ax.set(xlabel="Tiempo (s)", ylabel="Posición x (px)", title="Agentes vs. curva humana")
    ax.legend(fontsize=8)
    fig.savefig(R / "agentes_vs_humano.png", dpi=120, bbox_inches="tight")
    if diferencias:
        tabla = tabla.merge(pd.DataFrame(diferencias), on=["agente", "semilla"], how="left")
else:
    print("Aviso: no encontré curva_humana.joblib; corre 03_regresion.py para comparar con el humano.")

tabla.to_csv(R / "comparacion_agentes.csv", index=False)
print("Resultados por agente y semilla:\n", tabla.round(1).to_string(index=False))
print("\nPromedio por agente:\n",
      tabla.drop(columns="semilla").groupby("agente").mean().round(1).to_string())
print("\n(dif_media_vs_humano_px < 0 significa que el agente va por detrás del humano)")
print(f"Gráficos guardados en {R}")
