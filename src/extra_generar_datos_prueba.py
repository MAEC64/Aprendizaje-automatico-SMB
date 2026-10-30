"""OPCIONAL: genera un dataset de prueba con el MISMO formato que el real.

Sirve para comprobar que todos los scripts funcionan antes de descargar los
889 MB del dataset. Juega varios episodios con una política sencilla (con
algo de azar) y guarda cada fotograma como PNG con los bloques RAM, BP1 y
OUTCOME al final, imitando incluso el error de modo texto de Windows.

Uso:
    python src/extra_generar_datos_prueba.py --episodios 12
Los datos quedan en datos/prueba/. Bórralos antes de usar el dataset real:
    rm -rf datos/prueba
"""
import argparse
import random
from datetime import datetime

from PIL import Image

import comun

# Acciones del agente -> byte de botones del dataset
A_BYTE = {0: 0, 1: 4, 2: 4 | 128, 3: 4 | 16, 4: 4 | 16 | 128}


def bloque_texto(clave, valor):
    """Bloque tEXt tal como lo escribía el script Lua (CRC falso)."""
    contenido = clave.encode() + b"\x00" + valor
    crudo = len(contenido).to_bytes(4, "big") + b"tEXt" + contenido + b"\x00\x00\x00\x00"
    return crudo.replace(b"\n", b"\r\n")      # error del modo texto de Windows


def politica(ram, paso, rng):
    """Corre a la derecha y salta ante enemigos, obstáculos o huecos."""
    mx = comun.posicion_x(ram)
    peligro = comun.rango_enemigo(ram, mx) in (2, 3) or comun.tipo_terreno(ram, mx) > 0
    if rng.random() < 0.05:
        return rng.randrange(5)
    return 4 if peligro or (paso % 50) < 3 else 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodios", type=int, default=12)
    args = p.parse_args()

    rng = random.Random(0)
    env = comun.crear_entorno()
    sesion = "prueba01"
    for ep in range(1, args.episodios + 1):
        env.reset(seed=ep)
        fotos, f, terminado, previa, cola = [], 0, False, 3, []
        while not terminado:
            if not cola:                          # decide cada 4 fotogramas
                decision = politica(comun.ram_de(env), f // 4, rng)
                cola = comun.acciones_por_fotograma(decision, previa, comun.ram_de(env), 4)
                previa = decision
            accion = cola.pop(0)
            obs, r, term, trunc, info = env.step(accion)
            terminado = term or trunc
            f += 1
            fotos.append((f, accion, obs[::4, ::4].copy(), comun.ram_de(env).copy()))
        resultado = "win" if info["flag_get"] else "fail"
        carpeta = comun.DATOS / "prueba" / f"Prueba_{sesion}_e{ep}_{comun.NIVEL}_{resultado}"
        carpeta.mkdir(parents=True, exist_ok=True)
        fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for f, accion, img, ram in fotos:
            nombre = (f"Prueba_{sesion}_e{ep}_{comun.NIVEL}_f{f}_a{A_BYTE[accion]}_{fecha}"
                      f".{resultado}.png")
            ruta = carpeta / nombre
            Image.fromarray(img).save(ruta)
            with open(ruta, "ab") as fh:
                fh.write(bloque_texto("RAM", ram.tobytes()))
                fh.write(bloque_texto("BP1", bytes([A_BYTE[accion]])))
                fh.write(bloque_texto("OUTCOME", bytes([2 if resultado == "win" else 1])))
        print(f"Episodio {ep}: {resultado}, x máx = {info['x_pos']}, {len(fotos)} fotogramas")
    env.close()


if __name__ == "__main__":
    main()
