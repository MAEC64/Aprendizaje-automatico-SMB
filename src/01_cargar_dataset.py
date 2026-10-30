"""Paso 1: convierte los PNG del dataset (solo nivel 1-1) en una tabla.

Cada PNG trae, DESPUÉS del bloque IEND, tres bloques de texto añadidos por
el script de captura: RAM (2048 bytes), BP1 (botones) y OUTCOME.

Sobre el error de la RAM (issue #4 del repositorio): el script de captura
abría el archivo en modo texto de Windows, que convierte cada byte 0x0A
("\\n") en 0x0D 0x0A ("\\r\\n"). Por eso a veces la RAM mide más de 2048
bytes. Lo corregimos deshaciendo esa conversión y comprobamos el resultado
comparando el mundo/nivel guardado en la RAM con el del nombre del archivo.

Uso:
    python src/01_cargar_dataset.py
Resultado:
    resultados/fotogramas_1-1.parquet
"""
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import comun

PATRON = re.compile(
    r"^(?P<usuario>.+)_(?P<sesion>[a-z0-9]+)_e(?P<ep>\d+)_(?P<mundo>\d)-(?P<nivel>\d)"
    r"_f(?P<fotograma>\d+)_a(?P<accion>\d+)_(?P<fecha>[\d_-]+?)(?:\.(?P<resultado>win|fail))?\.png$"
)
MARCAS = {"RAM": b"tEXtRAM\x00", "BP1": b"tEXtBP1\x00", "OUTCOME": b"tEXtOUTCOME\x00"}


def fin_del_png(datos):
    """Posición justo después del bloque IEND, recorriendo los bloques del PNG."""
    p = 8  # firma del PNG
    while p + 8 <= len(datos):
        largo = int.from_bytes(datos[p:p + 4], "big")
        tipo = datos[p + 4:p + 8]
        p += 12 + largo  # largo + tipo + datos + CRC
        if tipo == b"IEND":
            return p
    raise ValueError("PNG sin bloque IEND")


def deshacer_modo_texto(raw, largo_esperado):
    if len(raw) == largo_esperado:
        return raw
    return raw.replace(b"\r\n", b"\n")


def leer_metadatos(ruta):
    datos = Path(ruta).read_bytes()
    cola = datos[fin_del_png(datos):]
    i_ram, i_bp1, i_out = (cola.find(m) for m in MARCAS.values())
    if min(i_ram, i_bp1, i_out) < 0:
        return None
    ini_ram = i_ram + len(MARCAS["RAM"])
    # el dato termina 8 bytes antes de la siguiente marca (CRC falso + largo)
    ram_cruda = cola[ini_ram:i_bp1 - 8]
    corregida = len(ram_cruda) != 2048
    ram = deshacer_modo_texto(ram_cruda, 2048)
    bp1 = deshacer_modo_texto(cola[i_bp1 + len(MARCAS["BP1"]):i_out - 8], 1)
    ini_out = i_out + len(MARCAS["OUTCOME"])
    outcome = cola[ini_out:ini_out + 1]
    if len(ram) != 2048 or len(bp1) != 1:
        return None
    return np.frombuffer(ram, dtype=np.uint8), bp1[0], (outcome[0] if outcome else 0), corregida


def buscar_archivos():
    """Busca las carpetas de episodios del nivel elegido dentro de datos/."""
    sufijo = f"_{comun.NIVEL}"
    archivos = []
    for carpeta, subcarpetas, nombres in os.walk(comun.DATOS):
        if sufijo + "_" in Path(carpeta).name + "_":
            archivos += [Path(carpeta) / n for n in nombres if n.endswith(".png")]
    return archivos


def main():
    archivos = buscar_archivos()
    if not archivos:
        raise SystemExit(f"No encontré PNG del nivel {comun.NIVEL} dentro de {comun.DATOS}. "
                         "¿Descomprimiste data-smb.7z ahí?")
    print(f"Encontré {len(archivos)} fotogramas del nivel {comun.NIVEL}.")

    filas, malos, incoherentes, corregidas = [], 0, 0, 0
    for ruta in tqdm(archivos, desc="Leyendo PNG"):
        m = PATRON.match(ruta.name)
        meta = leer_metadatos(ruta) if m else None
        if meta is None:
            malos += 1
            continue
        ram, bp1, outcome, corregida = meta
        corregidas += corregida
        mundo, nivel = int(m["mundo"]), int(m["nivel"])
        if ram[comun.MUNDO] + 1 != mundo or ram[comun.NIVEL_RAM] + 1 != nivel:
            incoherentes += 1
        accion = int(m["accion"])
        filas.append({
            "episodio": f"{m['usuario']}_{m['sesion']}_e{m['ep']}",
            "fotograma": int(m["fotograma"]),
            "t": int(m["fotograma"]) / comun.FPS,
            "accion": accion,
            "accion_bp1": int(bp1),
            "accion_agente": comun.accion_humana_a_indice(accion),
            "x": comun.posicion_x(ram),
            "en_aire": int(ram[comun.EN_AIRE] != 0),
            "velocidad": int(np.int8(ram[comun.VEL_X])),
            "estado": comun.estado(ram),
            "resultado": m["resultado"] or {1: "fail", 2: "win"}.get(outcome, "?"),
            "archivo": str(ruta.relative_to(comun.DATOS)),
        })

    df = pd.DataFrame(filas).sort_values(["episodio", "fotograma"]).reset_index(drop=True)
    df.to_parquet(comun.TABLA_FOTOGRAMAS, index=False)

    # Resumen para el reporte (lo usa 08_resumen_reporte.py)
    por_ep = df.groupby("episodio").resultado.first()
    resumen = {"archivos_encontrados": len(archivos), "filas": len(df), "ilegibles": malos,
               "ram_corregidas": int(corregidas), "incoherentes": incoherentes,
               "episodios": int(df.episodio.nunique()), "ganados": int((por_ep == "win").sum()),
               "fallidos": int((por_ep == "fail").sum()),
               "acciones_descartadas": int((df.accion_agente < 0).sum())}
    (comun.RESULTADOS / "resumen_carga.json").write_text(json.dumps(resumen, indent=2))

    # Fotograma de ejemplo para el reporte: episodio completado (si hay), cerca del segundo 5
    ejemplos = df[df.resultado == "win"] if (df.resultado == "win").any() else df
    fila = ejemplos.iloc[(ejemplos.fotograma - 300).abs().argmin()]
    shutil.copy(comun.DATOS / fila.archivo, comun.RESULTADOS / "ejemplo_fotograma.png")

    print(f"\nGuardé {len(df)} filas en {comun.TABLA_FOTOGRAMAS}")
    print(f"RAM corregidas (error de modo texto): {corregidas}")
    print(f"Archivos que no se pudieron leer: {malos}")
    print(f"Fotogramas cuyo mundo/nivel en RAM no coincide con el nombre: {incoherentes} "
          "(debería ser 0 o casi 0; si es alto, la RAM se está leyendo mal)")
    print(f"Coincidencia acción del nombre vs. BP1: {(df.accion == df.accion_bp1).mean():.1%}")
    resumen = df.groupby("episodio").agg(resultado=("resultado", "first"),
                                          fotogramas=("fotograma", "max"),
                                          x_max=("x", "max"))
    print("\nEpisodios encontrados:\n", resumen.to_string())


if __name__ == "__main__":
    main()
