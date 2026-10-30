"""Paso 4: construye la tabla Q inicial del agente B a partir de las partidas humanas.

Para cada estado s se calcula con qué frecuencia el humano eligió cada acción a,
y se usa Q0(s, a) = BONO * frecuencia(a | s). Así el agente B empieza
prefiriendo lo que haría una persona. Los estados que el humano nunca visitó
quedan en 0.

Uso:
    python src/04_tabla_q_humana.py --bono 10
Resultado:
    resultados/q_inicial_humana.npy  (matriz de 90 estados x 5 acciones)
"""
import argparse

import numpy as np
import pandas as pd

import comun

p = argparse.ArgumentParser()
p.add_argument("--bono", type=float, default=10.0,
               help="valor que recibe la acción favorita del humano (escala de la recompensa)")
args = p.parse_args()

df = pd.read_parquet(comun.TABLA_FOTOGRAMAS)
# El agente decide cada 4 fotogramas, así que tomamos 1 de cada 4 y solo acciones válidas
df = df[(df.fotograma % 4 == 0) & (df.accion_agente >= 0)]

conteos = np.zeros((comun.N_ESTADOS, comun.N_ACCIONES))
np.add.at(conteos, (df.estado.to_numpy(), df.accion_agente.to_numpy()), 1)

totales = conteos.sum(axis=1, keepdims=True)
frecuencias = np.divide(conteos, totales, out=np.zeros_like(conteos), where=totales > 0)
q0 = args.bono * frecuencias
np.save(comun.RESULTADOS / "q_inicial_humana.npy", q0)

visitados = int((totales > 0).sum())
print(f"Pares (estado, acción) usados: {len(df)}")
print(f"Estados con datos humanos: {visitados} de {comun.N_ESTADOS}")
print("\nAcción preferida por el humano en los estados más frecuentes:")
for s in np.argsort(-totales[:, 0])[:10]:
    if totales[s, 0] == 0:
        break
    a = int(q0[s].argmax())
    print(f"  {comun.describir_estado(s):55s} -> {comun.ACCIONES[a]} ({frecuencias[s, a]:.0%})")
print(f"\nGuardado en {comun.RESULTADOS / 'q_inicial_humana.npy'}")
