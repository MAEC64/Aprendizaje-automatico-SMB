"""Paso 2: análisis exploratorio de las partidas humanas del nivel 1-1.

Uso:
    python src/02_exploracion.py
Resultados (en resultados/):
    eda_curvas_avance.png, eda_velocidad.png, eda_acciones.png,
    eda_acciones_por_estado.csv
"""
import matplotlib

matplotlib.use("Agg")  # guarda imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd

import comun

df = pd.read_parquet(comun.TABLA_FOTOGRAMAS)
R = comun.RESULTADOS

# 1) Estadísticas descriptivas por episodio
ep = df.groupby("episodio").agg(resultado=("resultado", "first"), duracion_s=("t", "max"),
                                x_max=("x", "max"))
ep["velocidad_media_px_s"] = ep.x_max / ep.duracion_s
print("Resumen por episodio:\n", ep.round(1).to_string())
print("\nEstadísticas descriptivas (todos los fotogramas):\n",
      df[["t", "x", "velocidad"]].describe().round(1).to_string())

# 2) Curvas de avance: posición x contra tiempo (lo que modelará la regresión)
fig, ax = plt.subplots(figsize=(9, 5))
for nombre, g in df.groupby("episodio"):
    gana = g.resultado.iloc[0] == "win"
    ax.plot(g.t, g.x, color="tab:green" if gana else "tab:red", alpha=0.5, lw=1)
ax.set(xlabel="Tiempo (s)", ylabel="Posición x (px)",
       title="Avance humano en el nivel 1-1 (verde = completado, rojo = fallido)")
fig.savefig(R / "eda_curvas_avance.png", dpi=120, bbox_inches="tight")

# 3) Histograma de velocidades
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df.velocidad, bins=40, color="tab:blue")
ax.set(xlabel="Velocidad horizontal (RAM 0x0057)", ylabel="Fotogramas", title="Velocidad de Mario")
fig.savefig(R / "eda_velocidad.png", dpi=120, bbox_inches="tight")

# 4) Frecuencia de acciones (traducidas a las 5 del agente; -1 = descartada)
nombres = {-1: "descartada", **dict(enumerate(comun.ACCIONES))}
conteo = df.accion_agente.map(nombres).value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
conteo.plot.bar(ax=ax, color="tab:orange")
ax.set(ylabel="Fotogramas", title="Acciones del jugador humano")
fig.savefig(R / "eda_acciones.png", dpi=120, bbox_inches="tight")
print("\nAcciones del humano:\n", (conteo / len(df)).round(3).to_string())

# 5) Qué hace el humano en cada estado (base de la tabla Q inicial del agente B)
validas = df[df.accion_agente >= 0]
tabla = pd.crosstab(validas.estado, validas.accion_agente.map(nombres), normalize="index").round(2)
tabla.insert(0, "fotogramas", validas.estado.value_counts().sort_index())
tabla.insert(0, "descripcion", [comun.describir_estado(s) for s in tabla.index])
tabla.sort_values("fotogramas", ascending=False).to_csv(R / "eda_acciones_por_estado.csv")
print(f"\nEstados distintos visitados por el humano: {len(tabla)} de {comun.N_ESTADOS}")
print("Los 8 más frecuentes:\n",
      tabla.sort_values("fotogramas", ascending=False).head(8).to_string())
print(f"\nGráficos guardados en {R}")
