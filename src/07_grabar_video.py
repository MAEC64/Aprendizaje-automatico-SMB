"""Paso 7: graba un video MP4 del agente jugando con su tabla Q final (sin exploración).

Uso:
    python src/07_grabar_video.py --agente B --semilla 0
    python src/07_grabar_video.py --agente B --semilla 0 --ver   # además lo muestra en ventana
Resultado:
    resultados/video_agente_<A|B>_s<semilla>.mp4
"""
import argparse

import imageio.v2 as imageio
import numpy as np

import comun

p = argparse.ArgumentParser()
p.add_argument("--agente", choices=["A", "B"], required=True)
p.add_argument("--semilla", type=int, default=0)
p.add_argument("--repetir", type=int, default=4)
p.add_argument("--max-segundos", type=int, default=120)
p.add_argument("--ver", action="store_true")
args = p.parse_args()

q = np.load(comun.RESULTADOS / f"agente_{args.agente}_s{args.semilla}_q.npy")
env = comun.crear_entorno("human" if args.ver else None)
obs, info = env.reset(seed=args.semilla)
ram = comun.ram_de(env)
cuadros, fin, a_previa = [obs.copy()], False, 0
while not fin and len(cuadros) < args.max_segundos * comun.FPS:
    a = int(q[comun.estado(ram)].argmax())
    for a_fotograma in comun.acciones_por_fotograma(a, a_previa, ram, args.repetir):
        obs, r, term, trunc, info = env.step(a_fotograma)
        cuadros.append(obs.copy())
        if term or trunc:
            fin = True
            break
    a_previa = a
env.close()

salida = comun.RESULTADOS / f"video_agente_{args.agente}_s{args.semilla}.mp4"
imageio.mimwrite(salida, cuadros, fps=comun.FPS, macro_block_size=8)
print(f"Video guardado en {salida} ({len(cuadros) / comun.FPS:.1f} s). "
      f"Llegó a x = {info['x_pos']}{' y completó el nivel' if info.get('flag_get') else ''}.")
