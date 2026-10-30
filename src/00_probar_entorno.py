"""Paso de verificación: comprueba que el emulador, la RAM y el estado funcionan.

Uso:
    python src/00_probar_entorno.py          # sin ventana (rápido)
    python src/00_probar_entorno.py --ver    # abre una ventana con el juego
"""
import argparse
import time

import comun

p = argparse.ArgumentParser()
p.add_argument("--ver", action="store_true", help="mostrar el juego en una ventana")
args = p.parse_args()

env = comun.crear_entorno("human" if args.ver else None)
obs, info = env.reset(seed=0)
print("Imagen del juego:", obs.shape, "| RAM:", comun.ram_de(env).shape)

inicio, pasos, terminado = time.time(), 0, False
while not terminado:
    accion = 4 if pasos % 40 < 15 else 3          # correr y saltar de vez en cuando
    obs, r, term, trunc, info = env.step(accion)
    terminado = term or trunc
    if pasos % 30 == 0:
        s = comun.estado(comun.ram_de(env))
        print(f"paso {pasos:4d}  x={info['x_pos']:5d}  estado {s:2d}: {comun.describir_estado(s)}")
    pasos += 1

print(f"\nFin del episodio en x={info['x_pos']} tras {pasos} fotogramas "
      f"({pasos / (time.time() - inicio):.0f} fotogramas/s). Todo listo.")
env.close()
